document.addEventListener('DOMContentLoaded', async function () {
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

      // Fetch blog stats
      try {
        await fetchBlogStats();
      } catch (error) {
        console.error('Error fetching blog stats:', error);
      }

      // Fetch prediction stats
      await fetchPredictionStats();

      // Helper function to get start and end of current week
      function getWeekRange() {
        const now = new Date();
        const dayOfWeek = now.getDay() || 7; // Convert Sunday (0) to 7
        const startOfWeek = new Date(now);
        startOfWeek.setDate(now.getDate() - dayOfWeek + 1); // Start from Monday
        startOfWeek.setHours(0, 0, 0, 0);

        const endOfWeek = new Date(startOfWeek);
        endOfWeek.setDate(startOfWeek.getDate() + 6);
        endOfWeek.setHours(23, 59, 59, 999);

        return { start: startOfWeek, end: endOfWeek };
      }

      // Fetch and populate prediction stats
      async function fetchPredictionStats() {
        try {
          const response = await api.predictions.getAll();
          const predictions = response.data || [];
          const now = new Date();

          // Calculate week range for "this week" metrics
          const { start: weekStart, end: weekEnd } = getWeekRange();

          // Process predictions
          const totalPredictions = predictions.length;
          const predictionsThisWeek = predictions.filter(p => {
            const predDate = new Date(p.prediction_date || p.created_at);
            return predDate >= weekStart && predDate <= weekEnd;
          }).length;

          // Get predictions from last week for comparison
          const lastWeekStart = new Date(weekStart);
          lastWeekStart.setDate(weekStart.getDate() - 7);
          const lastWeekEnd = new Date(weekStart);
          lastWeekEnd.setDate(weekStart.getDate() - 1);

          const predictionsLastWeek = predictions.filter(p => {
            const predDate = new Date(p.prediction_date || p.created_at);
            return predDate >= lastWeekStart && predDate <= lastWeekEnd;
          }).length;

          // Calculate week-over-week change
          let weekOverWeekChange = 0;
          if (predictionsLastWeek > 0) {
            weekOverWeekChange = ((predictionsThisWeek - predictionsLastWeek) / predictionsLastWeek) * 100;
          } else if (predictionsThisWeek > 0) {
            weekOverWeekChange = 100; // 100% increase from 0
          }

          // Sort predictions by date to find the latest
          const sortedPredictions = [...predictions].sort((a, b) =>
            new Date(b.prediction_date || b.created_at) - new Date(a.prediction_date || a.created_at)
          );

          const latestPrediction = sortedPredictions[0];

          // Update UI with real data
          // Total Predictions Card
          document.querySelector('.stat-card.primary .card-text').textContent = totalPredictions;
          document.querySelector('.stat-card.primary .small span').textContent = `${predictionsThisWeek} this week`;

          // Last Prediction Card
          if (latestPrediction) {
            // Check multiple possible fields for prediction result
            let predictionValue = 'No';
            
            // Check different possible field names and data types
            if (latestPrediction.locust_present !== undefined) {
              predictionValue = latestPrediction.locust_present ? 'Yes' : 'No';
            } else if (latestPrediction.prediction_result !== undefined) {
              // Handle different possible formats of prediction_result
              const result = latestPrediction.prediction_result;
              if (typeof result === 'boolean') {
                predictionValue = result ? 'Yes' : 'No';
              } else if (typeof result === 'number') {
                predictionValue = result === 1 ? 'Yes' : 'No';
              } else if (typeof result === 'string') {
                predictionValue = result.toLowerCase().includes('yes') || 
                                result === '1' || 
                                result.toLowerCase().includes('present') ? 'Yes' : 'No';
              }
            } else if (latestPrediction.status !== undefined) {
              // Check status field as fallback
              const status = latestPrediction.status.toString().toLowerCase();
              if (status.includes('present') || status === '1' || status === 'yes') {
                predictionValue = 'Yes';
              }
            }
            
            // Update the UI with the prediction value
            const lastPredictionCard = document.querySelectorAll('.stat-card.primary')[1];
            lastPredictionCard.querySelector('.card-text').textContent = predictionValue;
            
            // Update trend indicator based on prediction result
            const trendIcon = lastPredictionCard.querySelector('i');
            const isPositive = predictionValue.toLowerCase() === 'yes';
            trendIcon.className = `bi ${isPositive ? 'bi-arrow-up-circle-fill text-danger' : 'bi-arrow-down-circle-fill text-success'} me-1`;
            lastPredictionCard.querySelector('.small span').textContent =
              isPositive ? 'Locusts detected' : 'No locusts detected';
          }

          // Last Updated Card
          if (latestPrediction) {
            const lastUpdate = latestPrediction.prediction_date || latestPrediction.created_at;
            const updateDate = new Date(lastUpdate);
            const options = { year: 'numeric', month: 'short', day: 'numeric' };

            document.querySelector('.stat-card.warning .card-text').textContent =
              updateDate.toLocaleDateString(undefined, options);
            document.querySelector('.stat-card.warning .small span').textContent =
              formatRelativeTime(updateDate);
          }

           // Model Accuracy Card: Show probability of the latest prediction
           if (predictions.length > 0 && latestPrediction && latestPrediction.probability !== undefined) {
             const probability = latestPrediction.probability;
             const accuracyPercent = (probability * 100).toFixed(2);
             document.querySelector('.stat-card.info .card-text').textContent = `${accuracyPercent}%`;

             // Update status text based on probability
             let statusText = 'Excellent performance';
             let statusClass = 'text-success';
             if (probability < 0.8) {
               statusText = 'Needs improvement';
               statusClass = 'text-warning';
             } else if (probability < 0.9) {
               statusText = 'Good performance';
               statusClass = 'text-info';
             }

             const statusIcon = document.querySelector('.stat-card.info i');
             statusIcon.className = `bi ${statusClass.includes('success') ? 'bi-check-circle-fill' : 'bi-exclamation-circle-fill'} ${statusClass} me-1`;
             document.querySelector('.stat-card.info .small span').textContent = statusText;
           }

           // Feedback Summary Card
           const feedbackSummaryText = document.getElementById('feedbackSummaryText');
           const feedbackSummarySubtext = document.getElementById('feedbackSummarySubtext');
           if (predictions.length > 0) {
             const correctCount = predictions.filter(p => p.feedback === 'correct').length;
             const feedbackCount = predictions.filter(p => p.feedback === 'correct' || p.feedback === 'incorrect').length;
             const percent = feedbackCount > 0 ? ((correctCount / feedbackCount) * 100).toFixed(1) : 0;
             feedbackSummaryText.textContent = `${correctCount} / ${feedbackCount} correct`;
             feedbackSummarySubtext.innerHTML = `<i class='bi bi-info-circle'></i> <span>${percent}% marked correct</span>`;
           } else {
             feedbackSummaryText.textContent = 'N/A';
             feedbackSummarySubtext.innerHTML = `<i class='bi bi-info-circle'></i> <span>No feedback yet</span>`;
           }

        } catch (error) {
          console.error('Error fetching prediction stats:', error);
          // Show error in console but don't break the UI
        }
      }

      async function fetchBlogStats() {
        console.log('1. Starting fetchBlogStats...');
        
        try {
          // Get current user and token
          const user = window.api?.auth?.getCurrentUser?.();
          const token = localStorage.getItem('token');
          
          if (!user?.id || !token) {
            console.warn('User not authenticated or no token found');
            return [];
          }
          
          console.log('2. Fetching blog posts using API wrapper');
          const response = await window.api.blog.getPosts();
          console.log('3. Successfully fetched blog posts:', response);
           
           // Update the blog count
           const blogCountElement = document.getElementById('blogCount');
           const blogTrendElement = document.getElementById('blogTrend');
           
           if (blogCountElement && blogTrendElement) {
             const count = Array.isArray(response) ? response.length : 0;
             blogCountElement.textContent = count;
             
             // Calculate trend
             const weekStart = new Date();
             weekStart.setDate(weekStart.getDate() - 7);
             
             const postsThisWeek = response.filter(post => {
               const postDate = new Date(post.date || post.created_at);
               return postDate >= weekStart;
             }).length;
             
             blogTrendElement.textContent = `${postsThisWeek} this week`;
             
             console.log('4. Updated blog count to:', count);
           } else {
             console.error('Blog stats elements not found');
           }
           
           return response;
         } catch (error) {
           console.error('Error in fetchBlogStats:', error);
           console.error('Error details:', {
             message: error.message,
             name: error.name,
             stack: error.stack
           });
           
           // Check if it's a 401 error (unauthorized)
           if (error?.response?.status === 401) {
             console.error('Unauthorized access. Token may be expired.');
             // Optionally, you could redirect to login here
             // window.location.href = '/login.html';
           }
           
           // Update UI to show error state
           const blogCountElement = document.getElementById('blogCount');
           const blogTrendElement = document.getElementById('blogTrend');
           
           if (blogCountElement) blogCountElement.textContent = '0';
           if (blogTrendElement) {
             blogTrendElement.innerHTML = '<i class="bi bi-exclamation-circle me-1"></i>Error loading';
           }
           return [];
         }
       }

      // Call the fetch functions
      fetchPredictionStats();
      fetchBlogStats();
      
      // Initialize charts
      initializeCharts();
      
      // Load recent predictions
      loadRecentPredictions();
      
      // Load recent activities
      loadRecentActivities();
      
      // Load environmental conditions
      loadEnvironmentalConditions();
    });

    // Initialize all charts
    let weeklyChart, monthlyChart, regionalChart;
    
    function initializeCharts() {
      // Weekly Trend Chart
      const weeklyCtx = document.getElementById('weeklyTrendChart').getContext('2d');
      weeklyChart = new Chart(weeklyCtx, {
        type: 'line',
        data: {
          labels: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
          datasets: [{
            label: 'Predictions',
            data: [0, 0, 0, 0, 0, 0, 0],
            borderColor: 'rgb(78, 115, 223)',
            backgroundColor: 'rgba(78, 115, 223, 0.1)',
            tension: 0.4,
            fill: true
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: {
              display: false
            }
          },
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

      // Monthly Overview Chart
      const monthlyCtx = document.getElementById('monthlyOverviewChart').getContext('2d');
      monthlyChart = new Chart(monthlyCtx, {
        type: 'bar',
        data: {
          labels: [],
          datasets: [{
            label: 'Locust Present',
            data: [],
            backgroundColor: 'rgba(231, 74, 59, 0.8)'
          }, {
            label: 'No Locust',
            data: [],
            backgroundColor: 'rgba(28, 200, 138, 0.8)'
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          scales: {
            x: {
              stacked: true
            },
            y: {
              stacked: true,
              beginAtZero: true
            }
          }
        }
      });

      // Regional Distribution Chart
      const regionalCtx = document.getElementById('regionalDistChart').getContext('2d');
      regionalChart = new Chart(regionalCtx, {
        type: 'doughnut',
        data: {
          labels: [],
          datasets: [{
            data: [],
            backgroundColor: [
              'rgba(78, 115, 223, 0.8)',
              'rgba(28, 200, 138, 0.8)',
              'rgba(246, 194, 62, 0.8)',
              'rgba(231, 74, 59, 0.8)',
              'rgba(54, 185, 204, 0.8)'
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

    // Load recent predictions
    async function loadRecentPredictions() {
      try {
        const response = await api.predictions.getAll();
        const predictions = response.data || [];
        
        // Update charts with real data
        updateChartsWithData(predictions);
        
        // Display recent predictions in table
        const tbody = document.getElementById('recentPredictionsTable');
        const recentPredictions = predictions.slice(0, 5); // Show only 5 most recent
        
        if (recentPredictions.length === 0) {
          tbody.innerHTML = `
            <tr>
              <td colspan="5" class="text-center py-4">
                <p class="text-muted mb-0">No predictions yet</p>
                <a href="predict.html" class="btn btn-sm btn-primary mt-2">Make Your First Prediction</a>
              </td>
            </tr>
          `;
          return;
        }
        
        tbody.innerHTML = recentPredictions.map(pred => {
          const date = new Date(pred.created_at);
          const hasLocust = pred.prediction_result === 1 || pred.prediction_result === '1';
          const badgeClass = hasLocust ? 'bg-danger' : 'bg-success';
          const badgeText = hasLocust ? 'Locust Present' : 'No Locust';
          
          // Feedback cell logic
          const feedbackCell = (() => {
            if (pred.feedback === 'correct') {
              return `<span class="badge bg-success"><i class="bi bi-hand-thumbs-up"></i> Marked correct</span>`;
            } else if (pred.feedback === 'incorrect') {
              return `<span class="badge bg-danger"><i class="bi bi-hand-thumbs-down"></i> Marked incorrect</span>`;
            } else {
              return `
                <button class="btn btn-link p-0 feedback-btn" data-id="${pred.id}" data-feedback="correct" title="Mark as correct">
                  <i class="bi bi-hand-thumbs-up fs-5 text-success"></i>
                </button>
                <button class="btn btn-link p-0 feedback-btn" data-id="${pred.id}" data-feedback="incorrect" title="Mark as incorrect">
                  <i class="bi bi-hand-thumbs-down fs-5 text-danger"></i>
                </button>
              `;
            }
          })();

          return `
            <tr>
              <td>${date.toLocaleDateString()}</td>
              <td>${pred.region_name || 'N/A'}</td>
              <td>${pred.country_name || 'N/A'}</td>
              <td><span class="badge ${badgeClass}">${badgeText}</span></td>
              <td>${feedbackCell}</td>
            </tr>
          `;
        }).join('');
        
        // Add feedback event handler for the table
        tbody.addEventListener('click', async function(e) {
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
            try {
              await api.predictions.submitFeedback(predictionId, feedback);
              // Optionally, reload the table to keep everything in sync
              // loadRecentPredictions();
            } catch (err) {
              // Optionally, show a toast or revert the cell if you want
            }
          }
        });
        
      } catch (error) {
        console.error('Error loading recent predictions:', error);
      }
    }

    // Update charts with real data
    function updateChartsWithData(predictions) {
      console.log("Updating charts with predictions:", predictions);
      
      // Weekly data
      const weekDays = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];
      const weekData = new Array(7).fill(0);
      
      // Get current date and set to start of day
      const today = new Date();
      today.setHours(0, 0, 0, 0);
      
      // Get the start of the current week (Monday)
      const currentWeekStart = new Date(today);
      currentWeekStart.setDate(today.getDate() - today.getDay() + (today.getDay() === 0 ? -6 : 1));
      currentWeekStart.setHours(0, 0, 0, 0);
      
      console.log("Current week starts on:", currentWeekStart.toDateString());
      
      predictions.forEach(pred => {
        // Use prediction_date if available, otherwise fall back to created_at
        const predDate = new Date(pred.prediction_date || pred.created_at);
        predDate.setHours(0, 0, 0, 0); // Normalize time part
        
        // Check if prediction is within the current week
        if (predDate >= currentWeekStart) {
          const dayDiff = Math.floor((predDate - currentWeekStart) / (1000 * 60 * 60 * 24));
          if (dayDiff >= 0 && dayDiff < 7) {
            weekData[dayDiff]++;
            console.log(`Added prediction to ${weekDays[dayDiff]} (day ${dayDiff}):`, predDate.toDateString());
          }
        }
      });
      
      console.log("Weekly data:", weekData);
      
      // Update weekly chart if it exists
      if (window.weeklyChart) {
        weeklyChart.data.labels = weekDays;
        weeklyChart.data.datasets[0].data = weekData;
        weeklyChart.update();
      }
      
      // Monthly data
      const monthlyData = {};
      predictions.forEach(pred => {
        const date = new Date(pred.prediction_date || pred.created_at);
        const monthKey = `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, '0')}`;
        if (!monthlyData[monthKey]) {
          monthlyData[monthKey] = { locust: 0, noLocust: 0 };
        }
        
        if (pred.prediction_result === 1 || pred.prediction_result === '1' || pred.prediction_result === true) {
          monthlyData[monthKey].locust++;
        } else {
          monthlyData[monthKey].noLocust++;
        }
      });
      
      const sortedMonths = Object.keys(monthlyData).sort().slice(-6); // Last 6 months
      if (window.monthlyChart) {
        monthlyChart.data.labels = sortedMonths.map(m => {
          const [year, month] = m.split('-');
          return new Date(year, month - 1).toLocaleDateString('en-US', { month: 'short', year: 'numeric' });
        });
        monthlyChart.data.datasets[0].data = sortedMonths.map(m => monthlyData[m].locust);
        monthlyChart.data.datasets[1].data = sortedMonths.map(m => monthlyData[m].noLocust);
        monthlyChart.update();
      }
      
      // Regional data
      const regionalData = {};
      predictions.forEach(pred => {
        const region = pred.region_name || 'Unknown';
        regionalData[region] = (regionalData[region] || 0) + 1;
      });
      
      if (window.regionalChart) {
        regionalChart.data.labels = Object.keys(regionalData);
        regionalChart.data.datasets[0].data = Object.values(regionalData);
        regionalChart.update();
      }
    }

    // View prediction details
    function viewPredictionDetails(predictionId) {
      // Redirect to reports page with the prediction ID
      window.location.href = `reports.html#prediction-${predictionId}`;
    }

    // Load recent activities
    async function loadRecentActivities() {
      try {
        // Fetch recent predictions for activity feed
        const response = await api.predictions.getAll();
        const predictions = response.data || [];
        
        // Sort by date and get the most recent 3
        const recentActivities = predictions
          .sort((a, b) => new Date(b.created_at || b.prediction_date) - new Date(a.created_at || a.prediction_date))
          .slice(0, 3);
        
        const activityFeed = document.getElementById('activityFeed');
        
        if (recentActivities.length === 0) {
          activityFeed.innerHTML = `
            <div class="text-center py-4">
              <i class="bi bi-inbox fs-1 text-muted d-block mb-2"></i>
              <p class="text-muted mb-0">No recent activity</p>
              <a href="predict.html" class="btn btn-sm btn-primary mt-2">Make Your First Prediction</a>
            </div>
          `;
          return;
        }
        
        // Generate activity items
        activityFeed.innerHTML = recentActivities.map(activity => {
          const date = new Date(activity.created_at || activity.prediction_date);
          const timeAgo = formatRelativeTime(date);
          const hasLocust = activity.prediction_result === 1 || activity.prediction_result === '1';
          
          // Determine activity type and styling
          let icon, iconClass, itemClass, message;
          
          if (hasLocust) {
            icon = 'bi-exclamation-circle-fill';
            iconClass = 'text-danger';
            itemClass = 'danger';
            message = `High risk prediction for <strong>${activity.region_name || 'Unknown'}, ${activity.country_name || 'Unknown'}</strong>`;
          } else {
            icon = 'bi-check-circle-fill';
            iconClass = 'text-success';
            itemClass = 'success';
            message = `Prediction completed for <strong>${activity.region_name || 'Unknown'}, ${activity.country_name || 'Unknown'}</strong>`;
          }
          
          return `
            <div class="activity-item ${itemClass}">
              <div class="d-flex align-items-start">
                <i class="bi ${icon} ${iconClass} me-2"></i>
                <div class="flex-grow-1">
                  <p class="mb-0">${message}</p>
                  <small class="text-muted">${timeAgo}</small>
                </div>
              </div>
            </div>
          `;
        }).join('');
        
      } catch (error) {
        console.error('Error loading recent activities:', error);
        document.getElementById('activityFeed').innerHTML = `
          <div class="alert alert-danger">
            <i class="bi bi-exclamation-triangle me-2"></i>
            Failed to load recent activities
          </div>
        `;
      }
    }

    // Helper function to format relative time
    function formatRelativeTime(date) {
      const now = new Date();
      const seconds = Math.floor((now - date) / 1000);
      
      const intervals = [
        { label: 'year', seconds: 31536000 },
        { label: 'month', seconds: 2592000 },
        { label: 'week', seconds: 604800 },
        { label: 'day', seconds: 86400 },
        { label: 'hour', seconds: 3600 },
        { label: 'minute', seconds: 60 }
      ];
      
      for (const interval of intervals) {
        const count = Math.floor(seconds / interval.seconds);
        if (count >= 1) {
          return `${count} ${interval.label}${count !== 1 ? 's' : ''} ago`;
        }
      }
      
      return 'Just now';
    }

    // Load environmental conditions
    async function loadEnvironmentalConditions() {
      try {
        const response = await api.predictions.getAll();
        const predictions = response.data || [];
        
        if (predictions.length === 0) {
          // Keep default values if no data
          return;
        }
        
        // Get recent predictions (last 30 days)
        const thirtyDaysAgo = new Date();
        thirtyDaysAgo.setDate(thirtyDaysAgo.getDate() - 30);
        
        const recentPredictions = predictions.filter(p => {
          const predDate = new Date(p.created_at || p.prediction_date);
          return predDate >= thirtyDaysAgo;
        });
        
        // Use all predictions if no recent ones
        const dataToUse = recentPredictions.length > 0 ? recentPredictions : predictions;
        
        // Calculate averages
        const avgTemp = dataToUse.reduce((sum, p) => sum + (p.temperature_celsius || 0), 0) / dataToUse.length;
        const avgMoisture = dataToUse.reduce((sum, p) => sum + (p.soil_moisture_percent || 0), 0) / dataToUse.length;
        const avgPrecipitation = dataToUse.reduce((sum, p) => sum + (p.precipitation_mm || 0), 0) / dataToUse.length;
        
        // Update the UI
        document.getElementById('currentTemp').textContent = `${avgTemp.toFixed(1)}°C`;
        document.getElementById('currentMoisture').textContent = `${avgMoisture.toFixed(0)}%`;
        document.getElementById('currentRain').textContent = `${avgPrecipitation.toFixed(1)}mm`;
        
        // Add dynamic styling based on values
        updateEnvironmentalIndicators(avgTemp, avgMoisture, avgPrecipitation);
        
        // Add a note about the data source
        const widget = document.querySelector('.weather-widget');
        if (!widget.querySelector('.data-source')) {
          const sourceNote = document.createElement('div');
          sourceNote.className = 'data-source text-center mt-3';
          sourceNote.innerHTML = `<small class="opacity-75">Average from ${dataToUse.length} recent prediction${dataToUse.length > 1 ? 's' : ''}</small>`;
          widget.appendChild(sourceNote);
        }
        
      } catch (error) {
        console.error('Error loading environmental conditions:', error);
        // Keep default values on error
      }
    }

    // Update environmental indicators with color coding
    function updateEnvironmentalIndicators(temp, moisture, precipitation) {
      // Temperature indicator (optimal: 25-35°C)
      const tempElement = document.getElementById('currentTemp');
      const tempParent = tempElement.parentElement.parentElement;
      if (temp >= 25 && temp <= 35) {
        tempParent.classList.add('optimal');
        tempElement.style.color = '#4ade80'; // Green
      } else if (temp < 20 || temp > 40) {
        tempElement.style.color = '#f87171'; // Red
      } else {
        tempElement.style.color = '#fbbf24'; // Yellow
      }
      
      // Soil Moisture indicator (optimal: 60-80%)
      const moistureElement = document.getElementById('currentMoisture');
      const moistureParent = moistureElement.parentElement.parentElement;
      if (moisture >= 60 && moisture <= 80) {
        moistureParent.classList.add('optimal');
        moistureElement.style.color = '#4ade80'; // Green
      } else if (moisture < 40 || moisture > 90) {
        moistureElement.style.color = '#f87171'; // Red
      } else {
        moistureElement.style.color = '#fbbf24'; // Yellow
      }
      
      // Precipitation indicator (context-dependent)
      const rainElement = document.getElementById('currentRain');
      if (precipitation > 50) {
        rainElement.style.color = '#60a5fa'; // Blue (high)
      } else if (precipitation < 10) {
        rainElement.style.color = '#fbbf24'; // Yellow (low)
      } else {
        rainElement.style.color = '#4ade80'; // Green (moderate)
      }
    }  
    
    // Check if SweetAlert2 is loaded
    if (typeof Swal === 'undefined') {
      console.error('SweetAlert2 failed to load. Please check your internet connection.');
    }