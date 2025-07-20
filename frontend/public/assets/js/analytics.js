document.addEventListener('DOMContentLoaded', function () {
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

      // Function to fetch and render analytics charts
      async function loadAnalyticsData() {
        try {
          // Fetch Prediction Summary data
          const summaryResponse = await api.analytics.getPredictionSummary();
          const summaryData = summaryResponse.data; // Assuming data is in response.data
          console.log('Prediction Summary Data:', summaryData);

          // Render Prediction Summary Chart (Pie Chart)
          const summaryCtx = document.getElementById('predictionSummaryChart').getContext('2d');
          new Chart(summaryCtx, {
            type: 'pie',
            data: {
              labels: ['No Locust', 'Locust Present'],
              datasets: [{
                data: [summaryData.no || 0, summaryData.yes || 0],
                backgroundColor: ['#1cc88a', '#f6c23e'], // Green for No, Yellow for Yes
                hoverBackgroundColor: ['#17a673', '#f4b636']
              }]
            },
            options: {
              responsive: true,
              plugins: {
                legend: {
                  position: 'top',
                },
                title: {
                  display: false, // Title is already in the card header
                }
              }
            }
          });

          // Fetch Predictions Over Time data
          const timeSeriesResponse = await api.analytics.getPredictionsOverTime();
          const timeSeriesData = timeSeriesResponse.data; // Assuming data is in response.data
          console.log('Predictions Over Time Data:', timeSeriesData);

          // Prepare data for Predictions Over Time Chart (Line Chart)
          const timeLabels = timeSeriesData.map(item => item.month_year);
          const timeData = timeSeriesData.map(item => item.count);

          // Render Predictions Over Time Chart (Line Chart)
          const timeSeriesCtx = document.getElementById('predictionsOverTimeChart').getContext('2d');
          new Chart(timeSeriesCtx, {
            type: 'line',
            data: {
              labels: timeLabels,
              datasets: [{
                label: 'Number of Predictions',
                data: timeData,
                borderColor: '#4e73df', // Primary color
                backgroundColor: 'rgba(78, 115, 223, 0.05)', // Light primary color fill
                tension: 0.3, // Smooth the line
                fill: true,
              }]
            },
            options: {
              responsive: true,
              scales: {
                y: {
                  beginAtZero: true,
                  ticks: {
                    // Ensure integers only
                    callback: function (value) {
                      if (Number.isInteger(value)) {
                        return value;
                      }
                    }
                  }
                },
                x: {
                  // Configuration for X-axis if needed
                }
              },
              plugins: {
                legend: {
                  display: false // Hide dataset label as it's clear from title
                },
                title: {
                  display: false
                }
              }
            }
          });

          // Fetch Predictions by Location data
          const locationResponse = await api.analytics.getPredictionsByLocation();
          const locationData = locationResponse.data; // Assuming data is in response.data
          console.log('Predictions by Location Data:', locationData);

          // Prepare data for Predictions by Location Chart (Grouped Bar Chart)
          const locationLabels = locationData.map(item => `${item.region}, ${item.country_name}`);
          const totalCounts = locationData.map(item => item.total_count);
          const positiveCounts = locationData.map(item => item.positive_count);

          // Render Predictions by Location Chart (Grouped Bar Chart)
          const locationCtx = document.getElementById('predictionsByLocationChart').getContext('2d');
          new Chart(locationCtx, {
            type: 'bar',
            data: {
              labels: locationLabels,
              datasets: [{
                label: 'Total Predictions',
                data: totalCounts,
                backgroundColor: '#4e73df', // Primary color
              }, {
                label: 'Locust Present Predictions',
                data: positiveCounts,
                backgroundColor: '#f6c23e', // Warning color
              }]
            },
            options: {
              responsive: true,
              scales: {
                y: {
                  beginAtZero: true,
                  ticks: {
                    callback: function (value) { if (Number.isInteger(value)) return value; }
                  }
                }
              },
              plugins: {
                legend: { position: 'top' },
                title: { display: false }
              }
            }
          });

          // Fetch Environmental Factors Summary data
          const factorsResponse = await api.analytics.getEnvironmentalFactorsSummary();
          const factorsData = factorsResponse.data; // Assuming data is in response.data
          console.log('Environmental Factors Summary Data:', factorsData);

          // Prepare data for Environmental Factors Chart (Grouped Bar Chart)
          const factorLabels = ['Average PPT', 'Average TMAX', 'Average Soil Moisture'];
          const noLocustData = [factorsData.no.avg_ppt, factorsData.no.avg_tmax, factorsData.no.avg_soil_moisture];
          const yesLocustData = [factorsData.yes.avg_ppt, factorsData.yes.avg_tmax, factorsData.yes.avg_soil_moisture];

          // Render Environmental Factors Chart (Grouped Bar Chart)
          const factorsCtx = document.getElementById('environmentalFactorsChart').getContext('2d');
          new Chart(factorsCtx, {
            type: 'bar',
            data: {
              labels: factorLabels,
              datasets: [{
                label: 'No Locust',
                data: noLocustData,
                backgroundColor: '#1cc88a', // Success color
              }, {
                label: 'Locust Present',
                data: yesLocustData,
                backgroundColor: '#f6c23e', // Warning color
              }]
            },
            options: {
              responsive: true,
              scales: {
                y: {
                  beginAtZero: true,
                }
              },
              plugins: {
                legend: { position: 'top' },
                title: { display: false }
              }
            }
          });

        } catch (error) {
          console.error('Error loading analytics data:', error);
          // Display an error message on the page if loading fails
          const analyticsContent = document.querySelector('main.main .container-fluid > .row'); // Target the row directly
          if (analyticsContent) {
            analyticsContent.innerHTML = `
              <div class="col-12">
                <div class="alert alert-danger" role="alert">
                  <h4 class="alert-heading">Error Loading Analytics</h4>
                  <p>Could not load analytics data. Please ensure the backend is running and try again.</p>
                  <hr>
                  <p class="mb-0">Details: ${error.message || 'Unknown error'}</p>
                </div>
              </div>
            `;
          }
        }
      }

      // Call the function to load and display analytics data
      loadAnalyticsData();

      // Initialize the interactive feedback table
      function initFeedbackTable() {
        const table = document.getElementById('recentFeedbackTable');
        const tbody = table.querySelector('tbody');
        const searchInput = document.getElementById('feedbackSearch');
        const filterSelects = document.querySelectorAll('.filter-select');
        const sortableHeaders = table.querySelectorAll('th[data-sort]');
        const prevPageBtn = document.getElementById('prev-page');
        const nextPageBtn = document.getElementById('next-page');
        const currentPageSpan = document.getElementById('current-page');
        const startItemSpan = document.getElementById('start-item');
        const endItemSpan = document.getElementById('end-item');
        const totalItemsSpan = document.getElementById('total-items');
        
        let allFeedbackData = [];
        let filteredData = [];
        let currentPage = 1;
        const rowsPerPage = 10;
        let sortColumn = 'prediction_date';
        let sortDirection = 'desc';
        
        // Fetch feedback data from the API
        async function fetchFeedbackData() {
          try {
            const res = await api.analytics.getFeedbackAnalytics();
            const data = res.data || res;
            
            // Store the raw feedback data
            allFeedbackData = data.recent_feedback || [];
            
            // Update summary cards
            document.getElementById('feedbackTotal').innerText = data.total_feedback || 0;
            document.getElementById('feedbackCorrectPct').innerText = (data.correct_pct || 0) + '%';
            document.getElementById('feedbackIncorrectPct').innerText = (data.incorrect_pct || 0) + '%';
            
            // Initialize charts
            initFeedbackCharts(data);
            
            // Process feedback data for the table
            processFeedbackData();
            
          } catch (error) {
            console.error('Error fetching feedback data:', error);
            Toast.fire({
              icon: 'error',
              title: 'Error',
              text: 'Failed to load feedback data.'
            });
          }
        }
        
        // Initialize feedback charts
        function initFeedbackCharts(data) {
          // Pie Chart: Feedback Distribution
          new Chart(document.getElementById('feedbackPieChart').getContext('2d'), {
            type: 'pie',
            data: {
              labels: ['Correct', 'Incorrect'],
              datasets: [{
                data: [data.correct_count || 0, data.incorrect_count || 0],
                backgroundColor: ['#1cc88a', '#e74a3b']
              }]
            },
            options: { responsive: true }
          });

          // Line Chart: Feedback Over Time
          new Chart(document.getElementById('feedbackLineChart').getContext('2d'), {
            type: 'line',
            data: {
              labels: (data.feedback_over_time || []).map(x => x.period),
              datasets: [
                {
                  label: 'Correct',
                  data: (data.feedback_over_time || []).map(x => x.correct || 0),
                  borderColor: '#1cc88a',
                  backgroundColor: 'rgba(28,200,138,0.1)',
                  fill: true
                },
                {
                  label: 'Incorrect',
                  data: (data.feedback_over_time || []).map(x => x.incorrect || 0),
                  borderColor: '#e74a3b',
                  backgroundColor: 'rgba(231,74,59,0.1)',
                  fill: true
                }
              ]
            },
            options: { responsive: true }
          });

          // Bar Chart: Feedback by Region
          new Chart(document.getElementById('feedbackRegionChart').getContext('2d'), {
            type: 'bar',
            data: {
              labels: (data.feedback_by_region || []).map(x => x.region),
              datasets: [
                {
                  label: 'Correct',
                  data: (data.feedback_by_region || []).map(x => x.correct || 0),
                  backgroundColor: '#1cc88a'
                },
                {
                  label: 'Incorrect',
                  data: (data.feedback_by_region || []).map(x => x.incorrect || 0),
                  backgroundColor: '#e74a3b'
                }
              ]
            },
            options: { responsive: true, scales: { y: { beginAtZero: true } } }
          });
        }
        
        // Process feedback data for the table
        function processFeedbackData() {
          // Transform data to match our table structure
          allFeedbackData = allFeedbackData.map(item => {
            // Use the exact same logic as reports.js for the Result column
            const locustPresent = item.locust_present;
            let resultText = '';
            let resultClass = '';
            if (locustPresent === 1 || locustPresent === '1') {
              resultText = 'Locust Present';
              resultClass = 'bg-danger';
            } else if (locustPresent === 0 || locustPresent === '0') {
              resultText = 'No Locust';
              resultClass = 'bg-success';
            } else {
              resultText = String(locustPresent);
              resultClass = 'bg-secondary';
            }
            return {
              prediction_date: item.date,
              region: item.region || 'Unknown',
              country_name: item.country || 'Unknown',
              resultText,
              resultClass,
              feedback: item.feedback || null
            };
          });
          
          filteredData = [...allFeedbackData];
          updateFilterOptions();
          applyFiltersAndSort();
        }
        
        // Update filter dropdown options based on available data
        function updateFilterOptions() {
          const regions = new Set();
          const countries = new Set();
          const results = new Set();
          const feedbacks = new Set();
          allFeedbackData.forEach(item => {
            if (item.region) regions.add(item.region);
            if (item.country_name) countries.add(item.country_name);
            if (item.resultText) results.add(item.resultText);
            if (item.feedback) feedbacks.add(item.feedback.charAt(0).toUpperCase() + item.feedback.slice(1));
          });
          // Update region filter
          const regionSelect = document.querySelector('select[data-column="1"]');
          updateSelectOptions(regionSelect, Array.from(regions).sort());
          // Update country filter
          const countrySelect = document.querySelector('select[data-column="2"]');
          updateSelectOptions(countrySelect, Array.from(countries).sort());
          // Update result filter
          const resultSelect = document.querySelector('select[data-column="3"]');
          // Always show 'All Results', 'Locust Present', 'No Locust' in this order
          resultSelect.innerHTML = '';
          const allOption = document.createElement('option');
          allOption.value = '';
          allOption.textContent = 'All Results';
          resultSelect.appendChild(allOption);
          ['Locust Present', 'No Locust'].forEach(val => {
            const option = document.createElement('option');
            option.value = val;
            option.textContent = val;
            resultSelect.appendChild(option);
          });
        }
        
        function updateSelectOptions(select, options) {
          // Keep the first option ("All...") and remove others
          while (select.options.length > 1) {
            select.remove(1);
          }
          
          // Add new options
          options.forEach(option => {
            if (option) {  // Skip null/undefined options
              const optionElement = document.createElement('option');
              optionElement.value = option;
              optionElement.textContent = option;
              select.appendChild(optionElement);
            }
          });
        }
        
        // Apply filters, search, and sort to the data
        function applyFiltersAndSort() {
          // Apply search
          const searchTerm = searchInput.value.toLowerCase();
          
          // Apply filters
          filteredData = allFeedbackData.filter(item => {
            // Search filter
            const matchesSearch = !searchTerm || 
              (item.region && item.region.toLowerCase().includes(searchTerm)) ||
              (item.country_name && item.country_name.toLowerCase().includes(searchTerm)) ||
              (item.feedback && item.feedback.toLowerCase().includes(searchTerm));
            
            // Column filters
            const matchesFilters = Array.from(filterSelects).every(select => {
              const columnIndex = parseInt(select.dataset.column);
              const filterValue = select.value;
              
              if (!filterValue) return true;
              
              let itemValue;
              switch (columnIndex) {
                case 1: itemValue = item.region || '-'; break;
                case 2: itemValue = item.country_name || '-'; break;
                case 3: itemValue = item.resultText; break;
                case 4: itemValue = item.feedback ? (item.feedback.charAt(0).toUpperCase() + item.feedback.slice(1)) : 'No Feedback'; break;
                default: return true;
              }
              
              return itemValue === filterValue;
            });
            
            return matchesSearch && matchesFilters;
          });
          
          // Apply sorting
          filteredData.sort((a, b) => {
            let aValue = a[sortColumn];
            let bValue = b[sortColumn];
            
            // Handle different data types for sorting
            if (sortColumn === 'prediction_date') {
              aValue = new Date(aValue).getTime();
              bValue = new Date(bValue).getTime();
            } else if (typeof aValue === 'string') {
              aValue = aValue ? aValue.toLowerCase() : '';
              bValue = bValue ? bValue.toLowerCase() : '';
            }
            
            if (aValue < bValue) return sortDirection === 'asc' ? -1 : 1;
            if (aValue > bValue) return sortDirection === 'asc' ? 1 : -1;
            return 0;
          });
          
          // Reset to first page when filters change
          currentPage = 1;
          renderTable();
        }
        
        // Render the table with pagination
        function renderTable() {
          const startIndex = (currentPage - 1) * rowsPerPage;
          const endIndex = Math.min(startIndex + rowsPerPage, filteredData.length);
          const pageData = filteredData.slice(startIndex, endIndex);
          
          // Clear existing rows
          tbody.innerHTML = '';
          
          // Add new rows
          pageData.forEach(item => {
            const row = document.createElement('tr');
            
            // Format date
            const date = new Date(item.prediction_date);
            const formattedDate = date.toLocaleDateString('en-US', {
              year: 'numeric',
              month: 'short',
              day: 'numeric',
              hour: '2-digit',
              minute: '2-digit'
            });
            // Format feedback
            const feedback = item.feedback ? 
              `<span class="badge bg-${item.feedback === 'correct' ? 'success' : 'danger'}">
                ${item.feedback.charAt(0).toUpperCase() + item.feedback.slice(1)}
              </span>` : 
              '<span class="badge bg-secondary">No Feedback</span>';
            row.innerHTML = `
              <td>${formattedDate}</td>
              <td>${item.region || '-'}</td>
              <td>${item.country_name || '-'}</td>
              <td><span class="badge ${item.resultClass}">${item.resultText}</span></td>
              <td>${feedback}</td>
            `;
            tbody.appendChild(row);
          });
          
          // Update pagination info
          updatePaginationInfo();
        }
        
        // Update pagination information
        function updatePaginationInfo() {
          const totalPages = Math.ceil(filteredData.length / rowsPerPage);
          const startItem = filteredData.length > 0 ? (currentPage - 1) * rowsPerPage + 1 : 0;
          const endItem = Math.min(currentPage * rowsPerPage, filteredData.length);
          
          startItemSpan.textContent = startItem;
          endItemSpan.textContent = endItem;
          totalItemsSpan.textContent = filteredData.length;
          currentPageSpan.textContent = currentPage;
          
          // Update button states
          prevPageBtn.disabled = currentPage === 1;
          nextPageBtn.disabled = currentPage >= totalPages;
        }
        
        // Event Listeners
        searchInput.addEventListener('input', () => {
          applyFiltersAndSort();
        });
        
        filterSelects.forEach(select => {
          select.addEventListener('change', () => {
            applyFiltersAndSort();
          });
        });
        
        sortableHeaders.forEach(header => {
          header.addEventListener('click', (e) => {
            // Ignore clicks on select elements or their children
            if (e.target.tagName === 'SELECT' || e.target.closest('select')) {
              return;
            }
            
            const newSortColumn = header.dataset.sort;
            
            // Toggle sort direction if clicking the same column
            if (sortColumn === newSortColumn) {
              sortDirection = sortDirection === 'asc' ? 'desc' : 'asc';
            } else {
              sortColumn = newSortColumn;
              sortDirection = 'asc';
            }
            
            // Update sort indicators
            sortableHeaders.forEach(h => {
              const icon = h.querySelector('i');
              if (h === header) {
                icon.className = `bi bi-arrow-${sortDirection === 'asc' ? 'up' : 'down'}-short`;
              } else {
                icon.className = 'bi bi-arrow-down-up';
              }
            });
            
            applyFiltersAndSort();
          });
        });
        
        prevPageBtn.addEventListener('click', () => {
          if (currentPage > 1) {
            currentPage--;
            renderTable();
          }
        });
        
        nextPageBtn.addEventListener('click', () => {
          const totalPages = Math.ceil(filteredData.length / rowsPerPage);
          if (currentPage < totalPages) {
            currentPage++;
            renderTable();
          }
        });
        
        // Initial load
        fetchFeedbackData();
      }
      
      // Call the initialization function
      initFeedbackTable();
    });
  

  function updateSelectOptions(select, options) {
    // Keep the first option ("All...") and remove others
    while (select.options.length > 1) {
      select.remove(1);
    }

    // Add new options
    options.forEach(option => {
      const optionElement = document.createElement('option');
      optionElement.value = option;
      optionElement.textContent = option;
      select.appendChild(optionElement);
    });
  }

  // Apply filters, search, and sort to the data
  function applyFiltersAndSort() {
    // Apply search
    const searchTerm = searchInput.value.toLowerCase();

    // Apply filters
    filteredData = allFeedbackData.filter(item => {
      // Search filter
      const matchesSearch = !searchTerm ||
        (item.region && item.region.toLowerCase().includes(searchTerm)) ||
        (item.country_name && item.country_name.toLowerCase().includes(searchTerm)) ||
        (item.feedback && item.feedback.toLowerCase().includes(searchTerm));

      // Column filters
      const matchesFilters = Array.from(filterSelects).every(select => {
        const columnIndex = parseInt(select.dataset.column);
        const filterValue = select.value;

        if (!filterValue) return true;

        let itemValue;
        switch (columnIndex) {
          case 1: itemValue = item.region || '-'; break;
          case 2: itemValue = item.country_name || '-'; break;
          case 3: itemValue = item.resultText; break;
          case 4: itemValue = item.feedback ? (item.feedback.charAt(0).toUpperCase() + item.feedback.slice(1)) : 'No Feedback'; break;
          default: return true;
        }

        return itemValue === filterValue;
      });

      return matchesSearch && matchesFilters;
    });

    // Apply sorting
    filteredData.sort((a, b) => {
      let aValue = a[sortColumn];
      let bValue = b[sortColumn];

      // Handle different data types for sorting
      if (sortColumn === 'prediction_date') {
        aValue = new Date(aValue).getTime();
        bValue = new Date(bValue).getTime();
      } else if (typeof aValue === 'string') {
        aValue = aValue.toLowerCase();
        bValue = bValue.toLowerCase();
      }

      if (aValue < bValue) return sortDirection === 'asc' ? -1 : 1;
      if (aValue > bValue) return sortDirection === 'asc' ? 1 : -1;
      return 0;
    });

    // Reset to first page when filters change
    currentPage = 1;
    renderTable();
  }

  // Render the table with pagination
  function renderTable() {
    const startIndex = (currentPage - 1) * rowsPerPage;
    const endIndex = Math.min(startIndex + rowsPerPage, filteredData.length);
    const pageData = filteredData.slice(startIndex, endIndex);

    // Clear existing rows
    tbody.innerHTML = '';

    // Add new rows
    pageData.forEach(item => {
      const row = document.createElement('tr');

      // Format date
      const date = new Date(item.prediction_date);
      const formattedDate = date.toLocaleDateString('en-US', {
        year: 'numeric',
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
      });

      // Get the exact model result value from reports.js logic
      const modelResult = item.prediction_result;
      let resultText = '';
      let resultClass = '';
      if (modelResult === 1 || modelResult === '1') {
        resultText = 'Locust Present';
        resultClass = 'bg-danger';
      } else if (modelResult === 0 || modelResult === '0') {
        resultText = 'No Locust';
        resultClass = 'bg-success';
      } else {
        resultText = String(modelResult);
        resultClass = 'bg-secondary';
      }
      // Format feedback
      const feedback = item.feedback ?
        `<span class="badge bg-${item.feedback === 'correct' ? 'success' : 'danger'}">
          ${item.feedback.charAt(0).toUpperCase() + item.feedback.slice(1)}
        </span>` :
        '<span class="badge bg-secondary">No Feedback</span>';

      row.innerHTML = `
        <td>${formattedDate}</td>
        <td>${item.region || '-'}</td>
        <td>${item.country_name || '-'}</td>
        <td><span class="badge ${resultClass}">${resultText}</span></td>
        <td>${feedback}</td>
      `;

      tbody.appendChild(row);
    });

    // Update pagination info
    updatePaginationInfo();
  }

  // Update pagination information
  function updatePaginationInfo() {
    const totalPages = Math.ceil(filteredData.length / rowsPerPage);
    const startItem = filteredData.length > 0 ? (currentPage - 1) * rowsPerPage + 1 : 0;
    const endItem = Math.min(currentPage * rowsPerPage, filteredData.length);

    startItemSpan.textContent = startItem;
    endItemSpan.textContent = endItem;
    totalItemsSpan.textContent = filteredData.length;
    currentPageSpan.textContent = currentPage;

    // Update button states
    prevPageBtn.disabled = currentPage === 1;
    nextPageBtn.disabled = currentPage >= totalPages;
  }

  // Event Listeners
  searchInput.addEventListener('input', () => {
    applyFiltersAndSort();
  });

  filterSelects.forEach(select => {
    select.addEventListener('change', () => {
      applyFiltersAndSort();
    });
  });

  sortableHeaders.forEach(header => {
    header.addEventListener('click', () => {
      const newSortColumn = header.dataset.sort;

      // Toggle sort direction if clicking the same column
      if (sortColumn === newSortColumn) {
        sortDirection = sortDirection === 'asc' ? 'desc' : 'asc';
      } else {
        sortColumn = newSortColumn;
        sortDirection = 'asc';
      }

      // Update sort indicators
      sortableHeaders.forEach(h => {
        const icon = h.querySelector('i');
        if (h === header) {
          icon.className = `bi bi-arrow-${sortDirection === 'asc' ? 'up' : 'down'}-short`;
        } else {
          icon.className = 'bi bi-arrow-down-up';
        }
      });

      applyFiltersAndSort();
    });
  });

  prevPageBtn.addEventListener('click', () => {
    if (currentPage > 1) {
      currentPage--;
      renderTable();
    }
  });

  nextPageBtn.addEventListener('click', () => {
    const totalPages = Math.ceil(filteredData.length / rowsPerPage);
    if (currentPage < totalPages) {
      currentPage++;
      renderTable();
    }
  });

  // Initial load
  fetchFeedbackData();
