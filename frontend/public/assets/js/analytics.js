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
    });