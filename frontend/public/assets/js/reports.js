document.addEventListener('DOMContentLoaded', function() {
    // Check login status
    if (!api.auth.isLoggedIn()) {
      window.location.href = '/login.html';
      return;
    }
    
    // Load user info if logged in
    const user = api.auth.getCurrentUser();
    if (!user) {
       window.location.href = '/login.html';
       return;
    }

    // Update user profile
    const initials = user.full_name.split(' ').map(n => n[0]).join('').toUpperCase();
    document.getElementById('userInitials').innerText = initials;
    document.getElementById('userName').innerText = user.full_name;
    // You can add user email if you want:
    // document.querySelector('.profile-card p').innerText = user.email;

    // Initialize charts
    initializeCharts();
    
    // Load user predictions
    loadUserPredictions();
    
    // Set up event listeners for filter changes
    document.getElementById('fromYear').addEventListener('change', applyFilters);
    document.getElementById('toYear').addEventListener('change', applyFilters);
    document.getElementById('regionFilter').addEventListener('change', applyFilters);
    document.getElementById('countryFilter').addEventListener('change', applyFilters);

    // You can add JavaScript here to fetch and display report data later
    console.log("Reports page loaded for user:", user.email);
    
    // Example of where you might fetch data:
    // async function loadReports() {
    //   try {
    //     const predictions = await api.predictions.getAll();
    //     console.log("Fetched predictions for report:", predictions);
    //     // Render predictions in a table in the #reportContent div
    //   } catch (error) {
    //     console.error("Error loading report data:", error);
    //     document.getElementById('reportContent').innerHTML = '<p class="text-danger">Error loading report data.</p>';
    //   }
    // }
    // loadReports();
  });

  // Function to load user's predictions
  async function loadUserPredictions() {
    const tbody = document.getElementById('predictionHistory');
    if (!tbody) {
      console.error('Table body element not found');
      return;
    }

    try {
      console.log('Loading user predictions...');
      tbody.innerHTML = `
        <tr>
          <td colspan="5" class="text-center py-4">
            <div class="spinner-border text-primary" role="status">
              <span class="visually-hidden">Loading...</span>
            </div>
            <div>Loading predictions...</div>
          </td>
        </tr>`;

      const response = await api.predictions.getAll();
      console.log('API Response:', response);

      // Handle the response format
      let predictions = [];
      if (response && response.status === 'success' && Array.isArray(response.data)) {
          predictions = response.data;
      } else if (response && response.error) {
          throw new Error(response.details || response.error);
      } else if (Array.isArray(response)) {
          // Fallback for old format
          predictions = response;
      }

      console.log('Processed predictions:', predictions);

      // Clear existing rows
      tbody.innerHTML = '';

      if (!predictions || predictions.length === 0) {
        const tr = document.createElement('tr');
        tr.innerHTML = `
          <td colspan="5" class="text-center py-4">
            <div class="text-muted">
              <i class="bi bi-inbox fs-1 d-block mb-2"></i>
              No prediction history found. Make your first prediction to see it here!
            </div>
            <button class="btn btn-sm btn-outline-primary mt-2" onclick="location.reload()">
              <i class="bi bi-arrow-clockwise"></i> Refresh
            </button>
          </td>
        `;
        tbody.appendChild(tr);
        return;
      }

      // Store all predictions for filtering/pagination
      allPredictions = predictions;
      currentPage = 1; // Reset to first page
      
      // Update statistics
      updateStatistics(predictions);
      
      // Update charts
      updateCharts(predictions);
      
      // Populate filter dropdowns
      populateFilterDropdowns(predictions);
      
      // Apply initial filters
      applyFilters();

      // Add event listeners to delete buttons
      document.querySelectorAll('.delete-prediction').forEach(button => {
        button.addEventListener('click', handleDeletePrediction);
      });

    } catch (error) {
      console.error('Error loading predictions:', error);
      const errorMessage = error.message || 'Unknown error occurred';
      const statusCode = error.status || 'N/A';

      // Log detailed error information
      console.error('Error details:', {
        message: errorMessage,
        status: statusCode,
        timestamp: new Date().toISOString()
      });

      // Update the UI with error message
      const tbody = document.getElementById('predictionHistory');
      if (tbody) {
        tbody.innerHTML = `
          <tr>
            <td colspan="5" class="text-center py-4">
              <div class="alert alert-danger">
                <h6 class="alert-heading">Error Loading Predictions</h6>
                <p class="mb-1">${errorMessage}</p>
                <p class="mb-0 small">Status: ${statusCode}</p>
              </div>
              <button class="btn btn-sm btn-outline-primary mt-2" onclick="loadUserPredictions()">
                <i class="bi bi-arrow-clockwise"></i> Retry
              </button>
            </td>
          </tr>
        `;
      }

      // Show error toast with more details
      Swal.fire({
        icon: 'error',
        title: 'Error Loading Data',
        html: `
          <div class="text-start">
            <p class="mb-2">Failed to load prediction history:</p>
            <code class="d-block bg-light p-2 mb-2">${errorMessage}</code>
            <p class="text-muted small mb-0">Status: ${statusCode}</p>
          </div>
        `,
        confirmButtonText: 'OK',
        confirmButtonColor: '#3085d6',
        allowOutsideClick: false
      });
    }
  }

  // Function to handle prediction deletion
  async function handleDeletePrediction(event) {
    event.stopPropagation(); // Prevent any parent click events
    const button = event.currentTarget;
    const row = button.closest('tr');
    const predictionId = button.dataset.id;

    if (!predictionId) {
      console.error('No prediction ID found');
      return;
    }

    // Save original button state
    const originalButtonHTML = button.innerHTML;
    const originalButtonWidth = button.offsetWidth;
    button.style.width = `${originalButtonWidth}px`;

    try {
      const result = await Swal.fire({
        title: 'Are you sure?',
        text: 'This will permanently delete this prediction.',
        icon: 'warning',
        showCancelButton: true,
        confirmButtonColor: '#d33',
        cancelButtonColor: '#3085d6',
        confirmButtonText: 'Yes, delete it!',
        cancelButtonText: 'Cancel',
        reverseButtons: true,
        allowOutsideClick: false
      });

      if (result.isDismissed) {
        return; // User cancelled the deletion
      }

      // Show loading state
      button.disabled = true;
      button.innerHTML = '<span class="spinner-border spinner-border-sm me-1" role="status" aria-hidden="true"></span>Deleting...';

      // Disable other action buttons in the same row
      const actionButtons = row.querySelectorAll('button');
      actionButtons.forEach(btn => {
        if (btn !== button) btn.disabled = true;
      });

      try {
        // Call the API to delete the prediction
        const response = await api.predictions.delete(predictionId);
        console.log('Delete response:', response);

        // Show success message
        const toast = Swal.mixin({
          toast: true,
          position: 'top-end',
          showConfirmButton: false,
          timer: 3000,
          timerProgressBar: true
        });

        await toast.fire({
          icon: 'success',
          title: 'Prediction deleted successfully'
        });

        // Refresh the predictions list which will also update the stats
        await loadUserPredictions();

      } catch (error) {
        console.error('Error deleting prediction:', error);

        // Show error message
        Swal.fire({
          icon: 'error',
          title: 'Error',
          text: error.message || 'Failed to delete prediction. Please try again.',
          confirmButtonColor: '#3085d6'
        });

        // Reset button state
        button.disabled = false;
        button.innerHTML = originalButtonHTML;

        // Re-enable other buttons
        const actionButtons = row.querySelectorAll('button');
        actionButtons.forEach(btn => {
          btn.disabled = false;
        });

        return; // Stop further execution
      }

    } catch (error) {
      console.error('Error deleting prediction:', error);

      // Reset button state
      button.disabled = false;
      button.innerHTML = originalButtonHTML;

      // Show error message
      Swal.fire({
        icon: 'error',
        title: 'Error',
        text: error.message || 'Failed to delete prediction. Please try again.',
        confirmButtonColor: '#3085d6'
      });
    }
  }

  // Handle view prediction
  async function handleViewPrediction(event) {
    const button = event.currentTarget;
    const predictionId = button.getAttribute('data-id');

    // Store original button content at the start
    const originalContent = button.innerHTML;

    try {
      // Show loading state
      button.disabled = true;
      button.innerHTML = '<span class="spinner-border spinner-border-sm me-1" role="status" aria-hidden="true"></span>Loading...';

      // Disable other action buttons in the same row during loading
      const row = button.closest('tr');
      if (row) {
        const actionButtons = row.querySelectorAll('button');
        actionButtons.forEach(btn => {
          if (btn !== button) btn.disabled = true;
        });
      }

      // Fetch the prediction details with the correct API prefix
      const response = await fetch(`${API_BASE_URL}/api/predictions/${predictionId}`, {
        headers: getAuthHeader()
      });

      const data = await response.json();

      if (!response.ok || data.status !== 'success') {
        console.error('API Error:', data);
        throw new Error(data.message || 'Failed to fetch prediction details');
      }

      const prediction = data.data;
      
      // Debug: Log the complete prediction object and its structure
      console.log('Complete prediction object:', prediction);
      console.log('All prediction properties:');
      Object.entries(prediction).forEach(([key, value]) => {
        console.log(`${key}:`, value, `(type: ${typeof value})`);
      });
      
      // Check if we have a status field that might contain the locust presence info
      const hasStatusField = 'status' in prediction;
      console.log('Has status field?', hasStatusField);
      if (hasStatusField) {
        console.log('Status value:', prediction.status, 'type:', typeof prediction.status);
      }

      // Format the data for display
      const formattedData = [
        { label: 'User', value: prediction.full_name || 'N/A', icon: 'person' },
        { label: 'Email', value: prediction.email || 'N/A', icon: 'envelope' },
        { label: 'Start Year', value: prediction.start_year || 'N/A', icon: 'calendar' },
        { label: 'Start Month', value: getMonthName(prediction.start_month) || 'N/A', icon: 'calendar-month' },
        { label: 'PPT', value: prediction.ppt ? `${prediction.ppt} mm` : 'N/A', icon: 'cloud-rain' },
        { label: 'TMAX', value: prediction.tmax ? `${prediction.tmax}°C` : 'N/A', icon: 'thermometer-high' },
        { label: 'Soil Moisture', value: prediction.soil_moisture || 'N/A', icon: 'moisture' }
      ];

      // Create HTML for the cards
      const cardsHtml = formattedData.map(item => `
        <div class="col-12 col-md-6 col-lg-4 mb-3">
          <div class="card h-100">
            <div class="card-body text-center">
              <div class="mb-2">
                <i class="bi bi-${item.icon} fs-2 text-primary"></i>
              </div>
              <h5 class="card-title mb-1">${item.label}</h5>
              <p class="card-text fs-5 fw-bold">${item.value}</p>
            </div>
          </div>
        </div>
      `).join('');

      // Show the SweetAlert with the prediction details
      Swal.fire({
        title: 'Prediction Details',
        html: `
          <div class="container-fluid">
            <div class="row g-3">
              ${cardsHtml}
            </div>
          </div>
        `,
        width: '80%',
        showCloseButton: true,
        showConfirmButton: false,
        customClass: {
          container: 'sweet-container',
          popup: 'sweet-popup',
          title: 'sweet-title',
          htmlContainer: 'sweet-html'
        }
      });

    } catch (error) {
      console.error('Error viewing prediction:', error);
      Swal.fire({
        icon: 'error',
        title: 'Error',
        text: 'Failed to load prediction details. Please try again.',
        confirmButtonColor: '#3085d6',
      });
    } finally {
      // Reset button state
      button.disabled = false;
      button.innerHTML = originalContent;
    }
  }

  // Helper function to get month name from number (1-12)
  function getMonthName(monthNumber) {
    if (!monthNumber) return 'N/A';
    const date = new Date(2000, monthNumber - 1, 1);
    return date.toLocaleString('default', { month: 'long' });
  }

  // Chart instances
  let timelineChart = null;
  let regionChart = null;

  // Pagination and search state
  let currentPage = 1;
  const rowsPerPage = 5; // Show 5 rows per page
  let allPredictions = [];
  let filteredPredictions = [];
  let selectedPredictions = new Set();
  let bulkSelectMode = false;

  // Initialize the page
  document.addEventListener('DOMContentLoaded', () => {
    loadUserPredictions();
    initializeCharts();
  });

  // Filter predictions by date and paginate
  function filterAndDisplayPredictions(searchDate) {
    let filteredPredictions = [...allPredictions];

    // Filter by date if search date is provided
    if (searchDate) {
      const searchDateObj = new Date(searchDate);
      const searchDateStr = searchDateObj.toISOString().split('T')[0];

      filteredPredictions = filteredPredictions.filter(prediction => {
        const predDate = new Date(prediction.created_at).toISOString().split('T')[0];
        return predDate === searchDateStr;
      });
    }

    // Update pagination
    updatePagination(filteredPredictions);

    // Display current page
    displayPredictionsPage(filteredPredictions, currentPage);
  }

  // Update pagination controls
  function updatePagination(predictions) {
    const totalPages = Math.ceil(predictions.length / rowsPerPage);
    const paginationEl = document.getElementById('pagination');

    if (!paginationEl) return;

    let paginationHTML = '';

    // Previous button
    paginationHTML += `
      <li class="page-item ${currentPage === 1 ? 'disabled' : ''}">
        <a class="page-link" href="#" data-page="${currentPage - 1}" aria-label="Previous">
          <span aria-hidden="true">&laquo;</span>
        </a>
      </li>`;

    // Page numbers
    for (let i = 1; i <= totalPages; i++) {
      paginationHTML += `
        <li class="page-item ${i === currentPage ? 'active' : ''}">
          <a class="page-link" href="#" data-page="${i}">${i}</a>
        </li>`;
    }

    // Next button
    paginationHTML += `
      <li class="page-item ${currentPage === totalPages || totalPages === 0 ? 'disabled' : ''}">
        <a class="page-link" href="#" data-page="${currentPage + 1}" aria-label="Next">
          <span aria-hidden="true">&raquo;</span>
        </a>
      </li>`;

    paginationEl.innerHTML = paginationHTML;

    // Add click handlers for pagination
    document.querySelectorAll('.page-link').forEach(link => {
      // Use mousedown instead of click to prevent any default behavior
      link.addEventListener('mousedown', (e) => {
        // Prevent default immediately
        e.preventDefault();
        e.stopPropagation();
        
        // Get the page number from the closest data-page attribute
        const pageElement = e.target.closest('[data-page]');
        if (!pageElement) return;
        
        const page = parseInt(pageElement.getAttribute('data-page'));
        if (page && page !== currentPage && page > 0 && page <= totalPages) {
          currentPage = page;
          displayPredictionsPage(filteredPredictions, currentPage);
          updatePagination(filteredPredictions);
        }
        
        // Prevent any further events
        e.stopImmediatePropagation();
        return false;
      }, { passive: false });
    });
  }

  // Display a page of predictions
  function displayPredictionsPage(predictions, page) {
    const tbody = document.getElementById('predictionHistory');
    if (!tbody) return;

    const start = (page - 1) * rowsPerPage;
    const end = start + rowsPerPage;
    const paginatedPredictions = predictions.slice(start, end);

    if (paginatedPredictions.length === 0) {
      const colspan = bulkSelectMode ? 6 : 5;
      tbody.innerHTML = `
        <tr>
          <td colspan="${colspan}" class="text-center py-4">
            <div class="text-muted">
              <i class="bi bi-inbox fs-1 d-block mb-2"></i>
              No predictions found
            </div>
          </td>
        </tr>`;
      return;
    }

    tbody.innerHTML = paginatedPredictions.map(prediction => {
      // Format the date
      const predictionDate = prediction.created_at || new Date().toISOString();
      const formattedDate = new Date(predictionDate).toLocaleDateString('en-US', {
        year: 'numeric',
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
      });

      // Determine status
      const hasLocust = prediction.prediction_result === 1 || prediction.prediction_result === '1';
      const statusClass = hasLocust ? 'bg-danger' : 'bg-success';
      const statusText = hasLocust ? 'Locust Present' : 'No Locust';

      // Add checkbox cell if in bulk select mode
      const checkboxCell = bulkSelectMode ? `
        <td>
          <input type="checkbox" class="table-checkbox prediction-checkbox" 
                 data-id="${prediction.id}" 
                 onchange="handleCheckboxChange(this)"
                 ${selectedPredictions.has(prediction.id) ? 'checked' : ''}>
        </td>
      ` : '';

      // Add feedback cell
      const feedbackCell = (() => {
        if (prediction.feedback === 'correct') {
          return `<span class="badge bg-success"><i class="bi bi-hand-thumbs-up"></i> Marked correct</span>`;
        } else if (prediction.feedback === 'incorrect') {
          return `<span class="badge bg-danger"><i class="bi bi-hand-thumbs-down"></i> Marked incorrect</span>`;
        } else {
          // Show two icon buttons only
          return `
            <button class="btn btn-link p-0 feedback-btn" data-id="${prediction.id}" data-feedback="correct" title="Mark as correct">
              <i class="bi bi-hand-thumbs-up fs-5 text-success"></i>
            </button>
            <button class="btn btn-link p-0 feedback-btn" data-id="${prediction.id}" data-feedback="incorrect" title="Mark as incorrect">
              <i class="bi bi-hand-thumbs-down fs-5 text-danger"></i>
            </button>
          `;
        }
      })();

      return `
        <tr>
          ${checkboxCell}
          <td>${formattedDate}</td>
          <td>${prediction.region_name || 'N/A'}</td>
          <td>${prediction.country_name || 'N/A'}</td>
          <td><span class="badge ${statusClass}">${statusText}</span></td>
          <td>${feedbackCell}</td>
          <td class="d-flex gap-2">
            <button class="btn btn-sm btn-outline-primary view-prediction" data-id="${prediction.id}">
              <i class="bi bi-eye me-1"></i>View
            </button>
            <button class="btn btn-sm btn-outline-danger delete-prediction" data-id="${prediction.id}">
              <i class="bi bi-trash me-1"></i>Delete
            </button>
          </td>
        </tr>`;
    }).join('');

    // Reattach event listeners to action buttons
    document.querySelectorAll('.delete-prediction').forEach(btn => {
      btn.addEventListener('click', handleDeletePrediction);
    });

    // Add event listeners to view buttons
    document.querySelectorAll('.view-prediction').forEach(btn => {
      btn.addEventListener('click', handleViewPrediction);
    });

    // Add event listeners to feedback buttons
    document.getElementById('predictionHistory').addEventListener('click', async function(e) {
      if (e.target.closest('.feedback-btn')) {
        const btn = e.target.closest('.feedback-btn');
        const predictionId = btn.dataset.id;
        const feedback = btn.dataset.feedback;

        // Instantly replace the feedback cell with a badge/text
        const cell = btn.closest('td');
        if (cell) {
          if (feedback === 'correct') {
            cell.innerHTML = '<span class="badge bg-success"><i class="bi bi-hand-thumbs-up"></i> Marked correct</span>';
          } else {
            cell.innerHTML = '<span class="badge bg-danger"><i class="bi bi-hand-thumbs-down"></i> Marked incorrect</span>';
          }
        }

        // Submit feedback in the background
        try {
          await api.predictions.submitFeedback(predictionId, feedback);
          // Optionally, reload the table to keep everything in sync
          // loadUserPredictions();
        } catch (err) {
          // Optionally, show a toast or revert the cell if you want
        }
      }
    });
  }

  // Update statistics cards
  function updateStatistics(predictions) {
    const total = predictions.length;
    const locustPresent = predictions.filter(p => p.prediction_result === 1 || p.prediction_result === '1').length;
    const noLocust = total - locustPresent;
    const detectionRate = total > 0 ? ((locustPresent / total) * 100).toFixed(1) : 0;

    document.getElementById('totalPredictions').textContent = total;
    document.getElementById('locustDetected').textContent = locustPresent;
    document.getElementById('noLocust').textContent = noLocust;
    document.getElementById('detectionRate').textContent = detectionRate + '%';
  }

  // Initialize charts
  function initializeCharts() {
    // Destroy existing chart instances if they exist
    if (timelineChart) {
      timelineChart.destroy();
    }
    if (regionChart) {
      regionChart.destroy();
    }
    
    // Timeline chart
    const timelineCtx = document.getElementById('timelineChart').getContext('2d');
    timelineChart = new Chart(timelineCtx, {
      type: 'line',
      data: {
        labels: [],
        datasets: [{
          label: 'Predictions',
          data: [],
          borderColor: 'rgb(75, 192, 192)',
          backgroundColor: 'rgba(75, 192, 192, 0.2)',
          tension: 0.1
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          y: {
            beginAtZero: true,
            ticks: {
              stepSize: 1
            }
          }
        }
      }
    });

    // Regional distribution chart
    const regionCtx = document.getElementById('regionChart').getContext('2d');
    regionChart = new Chart(regionCtx, {
      type: 'doughnut',
      data: {
        labels: [],
        datasets: [{
          data: [],
          backgroundColor: [
            'rgba(255, 99, 132, 0.8)',
            'rgba(54, 162, 235, 0.8)',
            'rgba(255, 205, 86, 0.8)',
            'rgba(75, 192, 192, 0.8)',
            'rgba(153, 102, 255, 0.8)',
            'rgba(255, 159, 64, 0.8)'
          ]
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            position: 'bottom'
          }
        }
      }
    });
  }

  // Update charts with data
  function updateCharts(predictions) {
    if (!timelineChart || !regionChart) {
      console.warn('Charts not initialized. Initializing now...');
      initializeCharts();
      
      // If still not initialized after trying, log error and return
      if (!timelineChart || !regionChart) {
        console.error('Failed to initialize charts');
        return;
      }
    }

    try {
      // Update timeline chart
      const monthlyData = {};
      predictions.forEach(p => {
        if (p && p.created_at) {
          const date = new Date(p.created_at);
          // Check if date is valid
          if (!isNaN(date.getTime())) {
            const monthKey = `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, '0')}`;
            monthlyData[monthKey] = (monthlyData[monthKey] || 0) + 1;
          }
        }
      });

      if (Object.keys(monthlyData).length > 0) {
        const sortedMonths = Object.keys(monthlyData).sort();
        
        if (timelineChart && timelineChart.data) {
          timelineChart.data.labels = sortedMonths.map(m => {
            const [year, month] = m.split('-');
            return new Date(year, month - 1).toLocaleDateString('en-US', { month: 'short', year: 'numeric' });
          });
          
          if (timelineChart.data.datasets && timelineChart.data.datasets[0]) {
            timelineChart.data.datasets[0].data = sortedMonths.map(m => monthlyData[m]);
            timelineChart.update();
          }
        }
      }

      // Update regional chart
      const regionalData = {};
      predictions.forEach(p => {
        if (p) {
          const region = p.region_name || 'Unknown';
          regionalData[region] = (regionalData[region] || 0) + 1;
        }
      });

      if (regionChart && regionChart.data && regionChart.data.datasets && regionChart.data.datasets[0]) {
        regionChart.data.labels = Object.keys(regionalData);
        regionChart.data.datasets[0].data = Object.values(regionalData);
        regionChart.update();
      }
    } catch (error) {
      console.error('Error updating charts:', error);
      // Try to reinitialize charts on error
      if (confirm('There was an error updating the charts. Would you like to try reloading the page?')) {
        window.location.reload();
      }
    }
  }

  // Store country-region mapping
  const countryRegionMap = new Map();
  
  // Update region dropdown based on selected country
  function updateRegionDropdown(selectedCountry) {
    const regionFilter = document.getElementById('regionFilter');
    
    // Clear existing options and add default
    regionFilter.innerHTML = '<option value="">All Regions</option>';
    
    // If a country is selected, show only its regions
    if (selectedCountry && countryRegionMap.has(selectedCountry)) {
      const regions = Array.from(countryRegionMap.get(selectedCountry)).sort();
      regions.forEach(region => {
        const option = document.createElement('option');
        option.value = region;
        option.textContent = region;
        regionFilter.appendChild(option);
      });
    }
  }
  
  // Initialize filter dropdowns
  document.addEventListener('DOMContentLoaded', function() {
    const countryFilter = document.getElementById('countryFilter');
    
    // Handle country selection change
    countryFilter.addEventListener('change', function() {
      updateRegionDropdown(this.value);
      applyFilters();
    });
    
    // Handle region selection change
    document.getElementById('regionFilter').addEventListener('change', applyFilters);
  });

  // Populate filter dropdowns with prediction data
  function populateFilterDropdowns(predictions) {
    // Clear existing data
    countryRegionMap.clear();
    const countryFilter = document.getElementById('countryFilter');
    
    // Clear existing country options except the first one
    while (countryFilter.options.length > 1) {
      countryFilter.remove(1);
    }
    
    // Build country-region mapping and collect unique countries
    const uniqueCountries = new Set();
    
    predictions.forEach(prediction => {
      if (!prediction.country_name || !prediction.region_name) return;
      
      const country = prediction.country_name;
      const region = prediction.region_name;
      
      // Add to unique countries
      uniqueCountries.add(country);
      
      // Add to country-region map
      if (!countryRegionMap.has(country)) {
        countryRegionMap.set(country, new Set());
      }
      countryRegionMap.get(country).add(region);
    });
    
    // Populate country dropdown
    Array.from(uniqueCountries).sort().forEach(country => {
      const option = document.createElement('option');
      option.value = country;
      option.textContent = country;
      countryFilter.appendChild(option);
    });
    
    // Initialize region dropdown
    updateRegionDropdown(countryFilter.value);
  }
    
    // Trigger filter update
    applyFilters();
  

  // Apply filters
  function applyFilters() {
    const fromYear = document.getElementById('fromYear').value;
    const toYear = document.getElementById('toYear').value;
    const region = document.getElementById('regionFilter').value;
    const country = document.getElementById('countryFilter').value;
    const status = document.getElementById('statusFilter').value;

    filteredPredictions = allPredictions.filter(p => {
      // Year range filter
      if (fromYear || toYear) {
        const predYear = new Date(p.created_at).getFullYear().toString();
        if (fromYear && predYear < fromYear) return false;
        if (toYear && predYear > toYear) return false;
      }

      // Region filter
      if (region && p.region_name !== region) return false;

      // Country filter
      if (country && p.country_name !== country) return false;

      // Status filter
      if (status !== '') {
        const hasLocust = p.prediction_result === 1 || p.prediction_result === '1';
        if (status === '1' && !hasLocust) return false;
        if (status === '0' && hasLocust) return false;
      }

      return true;
    });

    currentPage = 1;
    updatePagination(filteredPredictions);
    displayPredictionsPage(filteredPredictions, currentPage);
    updateStatistics(filteredPredictions);
    updateCharts(filteredPredictions);
    updateActiveFilterCount();
  }

  // Update active filter count
  function updateActiveFilterCount() {
    const fromYear = document.getElementById('fromYear').value;
    const toYear = document.getElementById('toYear').value;
    const region = document.getElementById('regionFilter').value;
    const country = document.getElementById('countryFilter').value;
    const status = document.getElementById('statusFilter').value;
    
    let activeCount = 0;
    if (fromYear) activeCount++;
    if (toYear) activeCount++;
    if (region) activeCount++;
    if (country) activeCount++;
    if (status) activeCount++;
    
    const activeFilterElement = document.getElementById('activeFilterCount');
    if (activeFilterElement) {
      activeFilterElement.textContent = `${activeCount} active`;
      // Show/hide based on active count
      activeFilterElement.style.display = activeCount > 0 ? 'inline-block' : 'none';
    }
  }

  // Clear filters
  function clearFilters() {
    document.getElementById('fromYear').value = '';
    document.getElementById('toYear').value = '';
    document.getElementById('regionFilter').value = '';
    document.getElementById('countryFilter').value = '';
    document.getElementById('statusFilter').value = '';
    applyFilters();
    updateActiveFilterCount();
  }

  // Export to CSV
  function exportToCSV() {
    const data = filteredPredictions.length > 0 ? filteredPredictions : allPredictions;
    
    if (data.length === 0) {
      Swal.fire('No Data', 'No predictions to export', 'info');
      return;
    }

    const headers = ['Date', 'Region', 'Country', 'Year', 'Month', 'Temperature', 'Precipitation', 'Soil Moisture', 'Locust Present'];
    const rows = data.map(p => [
      new Date(p.created_at).toLocaleDateString(),
      p.region_name || 'N/A',
      p.country_name || 'N/A',
      p.start_year || 'N/A',
      p.start_month || 'N/A',
      p.temperature_celsius || 'N/A',
      p.precipitation_mm || 'N/A',
      p.soil_moisture_percent || 'N/A',
      p.prediction_result === 1 || p.prediction_result === '1' ? 'Yes' : 'No'
    ]);

    let csvContent = headers.join(',') + '\n';
    rows.forEach(row => {
      csvContent += row.map(cell => `"${cell}"`).join(',') + '\n';
    });

    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `locust_predictions_${new Date().toISOString().split('T')[0]}.csv`;
    a.click();
    window.URL.revokeObjectURL(url);
  }

  // Export to PDF
  function exportToPDF() {
    const data = filteredPredictions.length > 0 ? filteredPredictions : allPredictions;
    
    if (data.length === 0) {
      Swal.fire('No Data', 'No predictions to export', 'info');
      return;
    }

    const { jsPDF } = window.jspdf;
    const doc = new jsPDF();

    // Add title
    doc.setFontSize(20);
    doc.text('Locust Prediction Report', 14, 22);

    // Add date
    doc.setFontSize(12);
    doc.text(`Generated: ${new Date().toLocaleDateString()}`, 14, 32);

    // Add statistics
    const total = data.length;
    const locustPresent = data.filter(p => p.prediction_result === 1 || p.prediction_result === '1').length;
    const detectionRate = ((locustPresent / total) * 100).toFixed(1);

    doc.text(`Total Predictions: ${total}`, 14, 42);
    doc.text(`Locust Detected: ${locustPresent}`, 14, 50);
    doc.text(`Detection Rate: ${detectionRate}%`, 14, 58);

    // Add table
    const tableData = data.map(p => [
      new Date(p.created_at).toLocaleDateString(),
      p.region_name || 'N/A',
      p.country_name || 'N/A',
      p.prediction_result === 1 || p.prediction_result === '1' ? 'Yes' : 'No'
    ]);

    doc.autoTable({
      head: [['Date', 'Region', 'Country', 'Locust Present']],
      body: tableData,
      startY: 70,
      theme: 'striped'
    });

    doc.save(`locust_report_${new Date().toISOString().split('T')[0]}.pdf`);
  }

  // Toggle bulk select mode
  function toggleBulkSelect() {
    bulkSelectMode = !bulkSelectMode;
    const checkboxHeaders = document.querySelectorAll('.bulk-select-header');
    const bulkActionsDiv = document.getElementById('bulkActions');
    
    if (bulkSelectMode) {
      // Show checkboxes
      checkboxHeaders.forEach(header => header.style.display = 'table-cell');
      
      // Update button text
      event.target.innerHTML = '<i class="bi bi-x-square"></i> Cancel Selection';
      event.target.classList.remove('btn-outline-primary');
      event.target.classList.add('btn-outline-secondary');
      
      // Clear any previous selections
      selectedPredictions.clear();
      updateBulkActionsUI();
      
      // Re-render the table with checkboxes
      displayPredictionsPage(filteredPredictions.length > 0 ? filteredPredictions : allPredictions, currentPage);
    } else {
      // Hide checkboxes
      checkboxHeaders.forEach(header => header.style.display = 'none');
      bulkActionsDiv.classList.remove('show');
      
      // Update button text
      event.target.innerHTML = '<i class="bi bi-check-square"></i> Select Multiple';
      event.target.classList.remove('btn-outline-secondary');
      event.target.classList.add('btn-outline-primary');
      
      // Clear selections
      selectedPredictions.clear();
      
      // Re-render the table without checkboxes
      displayPredictionsPage(filteredPredictions.length > 0 ? filteredPredictions : allPredictions, currentPage);
    }
  }

  // Toggle select all checkboxes
  function toggleSelectAll() {
    const selectAllCheckbox = document.getElementById('selectAll');
    const checkboxes = document.querySelectorAll('.prediction-checkbox');
    
    checkboxes.forEach(checkbox => {
      checkbox.checked = selectAllCheckbox.checked;
      const predictionId = parseInt(checkbox.dataset.id);
      
      if (selectAllCheckbox.checked) {
        selectedPredictions.add(predictionId);
      } else {
        selectedPredictions.delete(predictionId);
      }
    });
    
    updateBulkActionsUI();
  }

  // Handle individual checkbox selection
  function handleCheckboxChange(checkbox) {
    const predictionId = parseInt(checkbox.dataset.id);
    
    if (checkbox.checked) {
      selectedPredictions.add(predictionId);
    } else {
      selectedPredictions.delete(predictionId);
    }
    
    // Update select all checkbox state
    const allCheckboxes = document.querySelectorAll('.prediction-checkbox');
    const selectAllCheckbox = document.getElementById('selectAll');
    selectAllCheckbox.checked = selectedPredictions.size === allCheckboxes.length && allCheckboxes.length > 0;
    
    updateBulkActionsUI();
  }

  // Update bulk actions UI
  function updateBulkActionsUI() {
    const bulkActionsDiv = document.getElementById('bulkActions');
    const selectedCount = document.getElementById('selectedCount');
    
    selectedCount.textContent = selectedPredictions.size;
    
    if (selectedPredictions.size > 0) {
      bulkActionsDiv.classList.add('show');
    } else {
      bulkActionsDiv.classList.remove('show');
    }
  }

  // Bulk delete selected predictions
  async function bulkDelete() {
    if (selectedPredictions.size === 0) {
      Swal.fire('No Selection', 'Please select predictions to delete', 'info');
      return;
    }
    
    const result = await Swal.fire({
      title: 'Are you sure?',
      text: `You are about to delete ${selectedPredictions.size} prediction(s). This action cannot be undone.`,
      icon: 'warning',
      showCancelButton: true,
      confirmButtonColor: '#d33',
      cancelButtonColor: '#3085d6',
      confirmButtonText: 'Yes, delete them!',
      cancelButtonText: 'Cancel'
    });
    
    if (result.isConfirmed) {
      // Show loading
      Swal.fire({
        title: 'Deleting...',
        text: 'Please wait while we delete the selected predictions',
        allowOutsideClick: false,
        didOpen: () => {
          Swal.showLoading();
        }
      });
      
      let successCount = 0;
      let failCount = 0;
      
      // Delete each selected prediction
      for (const predictionId of selectedPredictions) {
        try {
          await api.predictions.delete(predictionId);
          successCount++;
        } catch (error) {
          console.error(`Failed to delete prediction ${predictionId}:`, error);
          failCount++;
        }
      }
      
      // Show result
      if (failCount === 0) {
        Swal.fire({
          icon: 'success',
          title: 'Deleted!',
          text: `Successfully deleted ${successCount} prediction(s).`,
          timer: 2000
        });
      } else {
        Swal.fire({
          icon: 'warning',
          title: 'Partial Success',
          text: `Deleted ${successCount} prediction(s). Failed to delete ${failCount} prediction(s).`
        });
      }
      
      // Reset bulk select mode and reload data
      bulkSelectMode = false;
      selectedPredictions.clear();
      document.querySelector('[onclick="toggleBulkSelect()"]').click();
      await loadUserPredictions();
    }
  }

  // Export selected predictions
  function exportSelected() {
    if (selectedPredictions.size === 0) {
      Swal.fire('No Selection', 'Please select predictions to export', 'info');
      return;
    }
    
    // Filter predictions to only selected ones
    const selectedData = allPredictions.filter(p => selectedPredictions.has(p.id));
    
    if (selectedData.length === 0) {
      Swal.fire('Error', 'Could not find selected predictions', 'error');
      return;
    }
    
    // Export as CSV
    const headers = ['Date', 'Region', 'Country', 'Year', 'Month', 'Temperature', 'Precipitation', 'Soil Moisture', 'Locust Present'];
    const rows = selectedData.map(p => [
      new Date(p.created_at).toLocaleDateString(),
      p.region_name || 'N/A',
      p.country_name || 'N/A',
      p.start_year || 'N/A',
      p.start_month || 'N/A',
      p.temperature_celsius || 'N/A',
      p.precipitation_mm || 'N/A',
      p.soil_moisture_percent || 'N/A',
      p.prediction_result === 1 || p.prediction_result === '1' ? 'Yes' : 'No'
    ]);
    
    let csvContent = headers.join(',') + '\n';
    rows.forEach(row => {
      csvContent += row.map(cell => `"${cell}"`).join(',') + '\n';
    });
    
    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `selected_predictions_${new Date().toISOString().split('T')[0]}.csv`;
    a.click();
    window.URL.revokeObjectURL(url);
    
    Swal.fire({
      icon: 'success',
      title: 'Export Complete',
      text: `Successfully exported ${selectedData.length} prediction(s)`,
      timer: 2000
    });
  }
  