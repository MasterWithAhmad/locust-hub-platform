// Make user available globally
let user = null;
let formSubmitted = false;

// Initialize the application when DOM is loaded
document.addEventListener("DOMContentLoaded", async function () {
  // Check login status
  if (!api.auth.isLoggedIn()) {
    window.location.href = "/login.html";
    return;
  }

  // Load user info if logged in
  user = api.auth.getCurrentUser();
  if (!user) {
    window.location.href = "/login.html";
    return;
  }

  // Update user profile in sidebar
  const initials = user.full_name
    .split(" ")
    .map((n) => n[0])
    .join("")
    .toUpperCase();
  document.getElementById("userInitials").innerText = initials;
  document.getElementById("userName").innerText = user.full_name;
  // The welcome message in the main content area for 'predict.html' is static,
  // but if you wanted to personalize it, you could:
  // document.getElementById('welcomeMessage').innerText = `Make a New Prediction, ${user.full_name}!`;

  // Fetch and populate region and country options
  try {
    const options = await api.options.getOptions();
    const regionList = document.getElementById("regionList");
    const countryList = document.getElementById("countryList");
    const startYearInput = document.getElementById("STARTYEAR");

    // Clear existing options (except the default)
    regionList.innerHTML = ""; // datalist doesn't have a default option
    countryList.innerHTML = ""; // datalist doesn't have a default option

    // Populate regions
    options.regions.forEach((region) => {
      const optionElement = document.createElement("option");
      optionElement.value = region;
      regionList.appendChild(optionElement);
    });

    // Populate countries
    options.countries.forEach((country) => {
      const optionElement = document.createElement("option");
      optionElement.value = country;
      countryList.appendChild(optionElement);
    });

    // Set the minimum year for the input field based on API options
    if (options.minYear) {
      startYearInput.min = options.minYear;
    }
    if (options.maxYear) {
      startYearInput.max = options.maxYear;
    }
  } catch (error) {
    console.error("Error fetching options:", error);
    Swal.fire({
      icon: "error",
      title: "Error loading options",
      text: "Could not load region and country options. Please try again later.",
    });
  }

  // Model selection is now hardcoded to Random Forest
  // The MODEL_NAME input is a hidden field in the HTML
});

// Function to validate all form fields
function validateForm() {
  let isValid = true;
  
  // Validate PPT (0.00-906.70)
  const pptInput = document.getElementById('PPT');
  if (!validateNumberInput(pptInput, 0, 906.70, 'Precipitation must be between 0.00mm and 906.70mm')) {
    isValid = false;
  }
  
  // Validate TMAX (-24 to 57.8)
  const tmaxInput = document.getElementById('TMAX');
  if (!validateNumberInput(tmaxInput, -24, 57.8, 'Temperature must be between -24°C and 57.8°C')) {
    isValid = false;
  }
  
  // Validate SOILMOISTURE (0-100)
  const soilMoistureInput = document.getElementById('SOILMOISTURE');
  if (!validateNumberInput(soilMoistureInput, 0, 100, 'Soil moisture must be between 0% and 100%')) {
    isValid = false;
  }
  
  // Check if any required fields are empty
  const requiredInputs = document.querySelectorAll('input[required]');
  requiredInputs.forEach(input => {
    if (!input.value.trim()) {
      input.classList.add('is-invalid');
      input.setCustomValidity('This field is required');
      input.reportValidity();
      isValid = false;
    }
  });
  
  return isValid;
}

async function handlePredictionSubmit(event) {
  // Prevent default form submission and propagation
  event.preventDefault();
  event.stopPropagation();
  
  // Validate form before proceeding
  if (!validateForm()) {
    // Scroll to the first invalid input
    const firstInvalid = document.querySelector('.is-invalid');
    if (firstInvalid) {
      firstInvalid.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
    return false;
  }
  
  // Get form reference from the event
  const form = event.target;
  
  // Manually collect form values to ensure all fields are captured
  const initialFormValues = {
    COUNTRYNAME: document.getElementById('COUNTRYNAME').value,
    REGION: document.getElementById('REGION').value,
    STARTYEAR: document.getElementById('STARTYEAR').value,
    STARTMONTH: document.getElementById('STARTMONTH').value,
    PPT: document.getElementById('PPT').value,
    TMAX: document.getElementById('TMAX').value,
    SOILMOISTURE: document.getElementById('SOILMOISTURE').value
  };
  
  const predictButton = form.querySelector('button[type="submit"]');
  if (!predictButton) {
    console.error('Submit button not found');
    return false;
  }

  // Get elements by ID directly since they are not children of the button
  const predictButtonText = document.getElementById('predictButtonText');
  const predictButtonLoading = document.getElementById('predictButtonLoading');
  const predictButtonLoadingText = document.getElementById('predictButtonLoadingText');

  // Set loading state
  predictButton.disabled = true;
  if (predictButtonText) predictButtonText.classList.add('d-none');
  if (predictButtonLoading) predictButtonLoading.classList.remove('d-none');
  if (predictButtonLoadingText) predictButtonLoadingText.classList.remove('d-none');

  // First, perform client-side validation
  if (!form.checkValidity()) {
    form.reportValidity();
    // Reset button state on validation failure
    predictButton.disabled = false;
    if (predictButtonText) predictButtonText.classList.remove('d-none');
    if (predictButtonLoading) predictButtonLoading.classList.add('d-none');
    if (predictButtonLoadingText) predictButtonLoadingText.classList.add('d-none');
    return false;
  }
  
  // Get values from the form data
  const startYear = parseInt(initialFormValues.STARTYEAR) || new Date().getFullYear();
  const startMonth = parseInt(initialFormValues.STARTMONTH) || 1;
  
  // Always use Random Forest model
  const modelName = 'random_forest.pkl';

  // Prepare prediction data with proper typing
  const predictionData = {
    REGION: initialFormValues.REGION?.trim() || '',
    COUNTRYNAME: initialFormValues.COUNTRYNAME?.trim() || '',
    STARTYEAR: startYear,
    STARTMONTH: startMonth,
    PPT: parseFloat(initialFormValues.PPT) || 0,
    TMAX: parseFloat(initialFormValues.TMAX) || 0,
    SOILMOISTURE: parseFloat(initialFormValues.SOILMOISTURE) || 0
  };

  console.log("Prediction data:", predictionData);

  try {
    const result = await api.predict(predictionData);
    console.log("Prediction API result:", result);

    // Reset button state with null checks
    if (predictButton) predictButton.disabled = false;
    if (predictButtonText) predictButtonText.classList.remove('d-none');
    if (predictButtonLoading) predictButtonLoading.classList.add('d-none');
    if (predictButtonLoadingText) predictButtonLoadingText.classList.add('d-none');

    // Repopulate form fields to prevent clearing
    Object.entries(initialFormValues).forEach(([key, value]) => {
      const input = form.elements[key];
      if (input) {
        if (input.type === 'checkbox' || input.type === 'radio') {
          input.checked = value === 'on';
        } else if (input.type === 'select-one' && input.id === 'MODEL_NAME') {
          // Special handling for model select
          input.value = modelName;
        } else {
          input.value = value || '';
        }
      }
    });

    // Show prediction result
    Swal.fire({
      title: result.prediction === "yes" ? "⚠️ YES ⚠️" : "✅ NO",
      html: `
                    <div style="text-align: center;">
                        <div style="font-size: 1.2em; margin: 15px 0;">
                            ${
                              result.prediction === "yes"
                                ? "High Risk of Locust Presence"
                                : "Low Risk of Locust Presence"
                            }
                        </div>
                        <div style="font-size: 1.1em; margin-bottom: 15px;">
                            Probability: <strong>${(
                              result.probability * 100
                            ).toFixed(1)}%</strong>
                        </div>
                    </div>
                `,
      icon: result.prediction === "yes" ? "warning" : "success",
      confirmButtonText: "OK",
      confirmButtonColor: "#198754",
      customClass: {
        title: "prediction-title",
        htmlContainer: "prediction-content",
        confirmButton: "btn btn-primary",
      },
      allowOutsideClick: false,
      allowEscapeKey: false,
    }).then((swalResult) => {
      if (swalResult.isConfirmed) {
        // After clicking OK, show detailed card
        const formattedDate = `${getMonthName(parseInt(initialFormValues.STARTMONTH))} ${
          initialFormValues.STARTYEAR
        }`;

        Swal.fire({
          title: "Prediction Details",
          html: `
                        <div class="container-fluid">
                            <div class="row g-3 mb-3">
                                <div class="col-sm-6">
                                    <div class="card bg-light border">
                                        <div class="card-body p-3">
                                            <div class="d-flex align-items-center">
                                                <i class="bi bi-globe text-primary me-2 fs-4"></i>
                                                <div>
                                                    <h6 class="card-title mb-1">Region</h6>
                                                    <p class="card-text mb-0">${
                                                      initialFormValues.REGION || 'N/A'
                                                    }</p>
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                                <div class="col-sm-6">
                                    <div class="card bg-light border">
                                        <div class="card-body p-3">
                                            <div class="d-flex align-items-center">
                                                <i class="bi bi-map text-primary me-2 fs-4"></i>
                                                <div>
                                                    <h6 class="card-title mb-1">Country</h6>
                                                    <p class="card-text mb-0">${
                                                      initialFormValues.COUNTRYNAME || 'N/A'
                                                    } (${result.matched_country || 'N/A'})</p>
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                                <div class="col-sm-6">
                                    <div class="card bg-light border">
                                        <div class="card-body p-3">
                                            <div class="d-flex align-items-center">
                                                <i class="bi bi-calendar3-fill text-primary me-2 fs-4"></i>
                                                <div>
                                                    <h6 class="card-title mb-1">Date</h6>
                                                    <p class="card-text mb-0">${formattedDate}</p>
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>

                            <h6 class="mb-2 fw-bold">Input Parameters</h6>
                            <div class="row g-3">
                                <div class="col-sm-4">
                                    <div class="card bg-light border">
                                        <div class="card-body p-3">
                                            <div class="d-flex align-items-center">
                                                <i class="bi bi-cloud-rain text-primary me-2 fs-4"></i>
                                                <div>
                                                    <h6 class="card-title mb-1">Precipitation</h6>
                                                    <p class="card-text mb-0">${
                                                      initialFormValues.PPT || 'N/A'
                                                    } mm</p>
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                                <div class="col-sm-4">
                                    <div class="card bg-light border">
                                        <div class="card-body p-3">
                                            <div class="d-flex align-items-center">
                                                <i class="bi bi-thermometer-high text-primary me-2 fs-4"></i>
                                                <div>
                                                    <h6 class="card-title mb-1">Max Temperature</h6>
                                                    <p class="card-text mb-0">${
                                                      initialFormValues.TMAX || 'N/A'
                                                    } °C</p>
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                                <div class="col-sm-4">
                                    <div class="card bg-light border">
                                        <div class="card-body p-3">
                                            <div class="d-flex align-items-center">
                                                <i class="bi bi-droplet text-primary me-2 fs-4"></i>
                                                <div>
                                                    <h6 class="card-title mb-1">Soil Moisture</h6>
                                                    <p class="card-text mb-0">${
                                                      initialFormValues.SOILMOISTURE || 'N/A'
                                                    }</p>
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>

                            <div class="alert ${
                              result.prediction === "yes"
                                ? "alert-warning"
                                : "alert-success"
                            } d-flex align-items-center mt-3" role="alert">
                                <i class="bi ${
                                  result.prediction === "yes"
                                    ? "bi-exclamation-triangle-fill"
                                    : "bi-check-circle-fill"
                                } me-2 fs-4"></i>
                                <div>
                                    <strong>Prediction:</strong> ${(
                                      result.probability * 100
                                    ).toFixed(
                                      1
                                    )}% Probability of locust presence
                                </div>
                            </div>
                        </div>
                    `,
          showCancelButton: true,
          confirmButtonText: '<i class="bi bi-save"></i> SAVE PREDICTION',
          cancelButtonText: '<i class="bi bi-x-lg"></i> CLOSE',
          confirmButtonColor: "#198754",
          cancelButtonColor: "#6c757d",
          reverseButtons: true,
          focusConfirm: false,
          showCloseButton: true,
          showClass: {
            popup: "animate__animated animate__fadeInDown animate__faster",
          },
          hideClass: {
            popup: "animate__animated animate__fadeOutUp animate__faster",
          },
          customClass: {
            popup: "prediction-details-modal",
          },
        }).then((saveResult) => {
          if (saveResult.isConfirmed) {
            // Format data for saving
            const predictionToSave = {
              region_name: initialFormValues.REGION.trim().toUpperCase(),
              country_name: initialFormValues.COUNTRYNAME.trim().toUpperCase(),
              start_year: parseInt(initialFormValues.STARTYEAR),
              start_month: parseInt(initialFormValues.STARTMONTH),
              precipitation_mm: parseFloat(initialFormValues.PPT),
              temperature_celsius: parseFloat(initialFormValues.TMAX),
              soil_moisture_percent: parseFloat(initialFormValues.SOILMOISTURE),
              prediction_result: result.probability > 0.5 ? 1 : 0,
              probability: result.probability,
            };

            // Save the prediction
            api.predictions
              .save(predictionToSave)
              .then(() => {
                Swal.fire({
                  title: "Saved!",
                  text: "Your prediction has been saved.",
                  icon: "success",
                  confirmButtonColor: "#198754",
                }).then(() => {
                  // If prediction is YES, ask if user wants to create an event/blog post
                  if (result.prediction === "yes") {
                    Swal.fire({
                      title: "Create Event/Blog Post?",
                      text: "Would you like to share this prediction as a public event or blog post?",
                      icon: "question",
                      showCancelButton: true,
                      confirmButtonText: "Yes, Create Blog",
                      cancelButtonText: "No",
                      confirmButtonColor: "#4e73df",
                      cancelButtonColor: "#6c757d",
                      reverseButtons: true,
                    }).then((blogResult) => {
                      if (blogResult.isConfirmed) {
                        // Show a form for blog post
                        showBlogModal({
                          user,
                          region: initialFormValues.REGION,
                          country: initialFormValues.COUNTRYNAME,
                          onPublish: async (blogData) => {
                            // Prepare form data for image upload
                            const fd = new FormData();
                            
                            // Function to clean HTML content
                            const cleanHtmlContent = (html) => {
                              if (!html) return '';
                              
                              // Create a temporary div to parse the HTML
                              const tempDiv = document.createElement('div');
                              tempDiv.innerHTML = html;
                              
                              // Remove any script tags and event handlers for security
                              const scripts = tempDiv.getElementsByTagName('script');
                              while (scripts[0]) {
                                scripts[0].parentNode.removeChild(scripts[0]);
                              }
                              
                              // Remove any style attributes that might contain dangerous content
                              const allElements = tempDiv.getElementsByTagName('*');
                              for (let el of allElements) {
                                el.removeAttribute('style');
                                el.removeAttribute('onclick');
                                el.removeAttribute('onload');
                                el.removeAttribute('onerror');
                              }
                              
                              // Return the cleaned HTML
                              return tempDiv.innerHTML;
                            };
                            
                            // Clean the content before sending
                            const cleanedContent = cleanHtmlContent(blogData.content);
                            
                            fd.append("title", blogData.title);
                            fd.append("content", cleanedContent);
                            fd.append("tags", blogData.tags);
                            fd.append("region", initialFormValues.REGION);
                            fd.append("country", initialFormValues.COUNTRYNAME);
                            fd.append("date", new Date().toISOString());
                            fd.append("author", user.full_name);
                            fd.append("user_id", user.id || "");
                            
                            if (blogData.imageFile) {
                              fd.append("image", blogData.imageFile);
                            }
                            Swal.showLoading();
                            // Get the JWT token from localStorage
                            const token = localStorage.getItem('token');
                            if (!token) {
                                throw new Error('Authentication required. Please log in again.');
                            }
                            
                            fetch("/api/blogposts", {
                              method: "POST",
                              headers: {
                                'Authorization': `Bearer ${token}`
                              },
                              body: fd,
                            })
                              .then(async (res) => {
                                const data = await res.json();
                                if (!res.ok) {
                                  // If response is not ok, throw an error with the message from the server
                                  const error = new Error(data.error || 'Failed to publish blog post');
                                  error.response = data;
                                  throw error;
                                }
                                return data;
                              })
                              .then((data) => {
                                console.log('Blog post created successfully:', data);
                                Swal.fire({
                                  title: "Published!",
                                  text: "Your event/blog post is now public.",
                                  icon: "success",
                                  confirmButtonColor: "#198754",
                                  showCancelButton: true,
                                  confirmButtonText: "View Blogs",
                                  cancelButtonText: "Close",
                                }).then((r) => {
                                  if (r.isConfirmed) {
                                    window.location.href = "blog-management.html";
                                  }
                                });
                              })
                              .catch((error) => {
                                console.error('Error creating blog post:', error);
                                Swal.fire({
                                  title: "Error",
                                  html: `Failed to publish blog post.<br><small class="text-muted">${error.message}</small>`,
                                  icon: "error",
                                  confirmButtonColor: "#dc3545",
                                });
                              });
                          },
                          onPreview: () => {},
                        });
                      }
                    });
                  }
                });
              })
              .catch((error) => {
                console.error("Error saving prediction:", error);
                Swal.fire({
                  title: "Error",
                  text: "Failed to save prediction. Please try again.",
                  icon: "error",
                  confirmButtonColor: "#dc3545",
                });
              });
          }
        });
      }
    });

    // Removed form reset to keep input values after submission
  } catch (error) {
    console.error("Prediction error:", error);
    // Hide loading state
    predictButton.disabled = false;
    predictButtonText.style.display = "inline-block";
    predictButtonLoading.style.display = "none";
    predictButtonLoadingText.style.display = "none";

    let errorMessage = "An error occurred during prediction.";
    if (error.response && error.response.data && error.response.data.error) {
      errorMessage = error.response.data.error;
      if (error.response.data.suggestions) {
        errorMessage +=
          "\nSuggestions: " + error.response.data.suggestions.join(", ");
      }
    }

    Swal.fire({
      icon: "error",
      title: "Prediction Failed",
      text: errorMessage,
    });
  }

  return false; // Prevent default form submission
}

// Helper function to get month name from number (1-12)
function getMonthName(monthNumber) {
  if (!monthNumber) return "N/A";
  const date = new Date(2000, monthNumber - 1, 1);
  return date.toLocaleString("default", { month: "long" });
}

// Global variables to store valid options
let validCountries = [];
let validRegions = [];
// countryRegionMap is now imported from country_region_map.js

// Updates the region datalist based on selected country
function updateRegionListForCountry(country) {
  const regionList = document.getElementById("regionList");
  regionList.innerHTML = "";
  const regions = countryRegionMap[country.trim().toUpperCase()];
  if (regions) {
    regions.forEach((region) => {
      const option = document.createElement("option");
      option.value = region;
      regionList.appendChild(option);
    });
  }
  // If no regions, datalist stays empty, but user can still enter any region
}

// Function to load valid countries and regions
async function loadValidOptions() {
  try {
    const response = await fetch("/api/options");
    if (!response.ok) {
      throw new Error("Failed to load options");
    }
    const data = await response.json();
    // Replace Somaliland with Somalia in the countries list
    validCountries = data.countries.map((c) => {
      const trimmed = c.trim().toUpperCase();
      return trimmed === "SOMALILAND" ? "SOMALIA" : trimmed;
    });
    validRegions = data.regions.map((r) => r.trim().toUpperCase());

    // Populate datalists
    const countryList = document.getElementById("countryList");
    const regionList = document.getElementById("regionList");

    // Create options for country list
    validCountries.forEach((country) => {
      const option = document.createElement("option");
      option.value = country;
      option.textContent = country === "SOMALIA" ? "Somalia" : country;
      countryList.appendChild(option);
    });

    // Get the country input element
    const countryInput = document.getElementById("COUNTRYNAME");

    // Add event listener to handle input changes
    countryInput.addEventListener("input", function () {
      let value = this.value.trim().toUpperCase();
      // Normalize Somalia variants
      if (
        [
          "SOMALIA",
          "SOMALILAND",
          "SOMALIA REPUBLIC",
          "SOMALI REPUBLIC",
        ].includes(value)
      ) {
        value = "SOMALIA";
        this.value = "Somalia";
      }
      // Update region datalist suggestions based on selected country
      console.log(
        "Country input (normalized):",
        value,
        countryRegionMap[value]
      );
      updateRegionListForCountry(value);
    });

    // Add event listener to handle selection from dropdown
    countryInput.addEventListener("change", function () {
      let value = this.value.trim().toUpperCase();
      // Normalize Somalia variants
      if (
        [
          "SOMALIA",
          "SOMALILAND",
          "SOMALIA REPUBLIC",
          "SOMALI REPUBLIC",
        ].includes(value)
      ) {
        value = "SOMALIA";
        this.value = "Somalia";
      }
      // Update region datalist suggestions based on selected country
      console.log(
        "Country change (normalized):",
        value,
        countryRegionMap[value]
      );
      updateRegionListForCountry(value);
    });

    // Initially, show all regions or leave region datalist empty
    regionList.innerHTML = "";
  } catch (error) {
    console.error("Error loading options:", error);
    Swal.fire({
      icon: "error",
      title: "Error",
      text: "Failed to load country and region options. Please refresh the page.",
    });
  }
}

// Function to validate dropdown/select inputs against allowed values
function validateInput(input, validValues, fieldName) {
  const value = input.value.trim().toUpperCase();

  // Special case for Somalia/Somaliland
  if (fieldName === "country" && value === "SOMALIA") {
    input.value = "SOMALILAND";
    input.setCustomValidity("");
    input.reportValidity();
    return;
  }

  if (value && !validValues.includes(value)) {
    input.setCustomValidity(`Please select a valid ${fieldName} from the list`);
  } else {
    input.setCustomValidity("");
  }
  input.reportValidity();
}

// Function to validate number inputs with min/max range
function validateNumberInput(input, min, max, errorMessage) {
  const value = parseFloat(input.value);
  const isValid = !isNaN(value) && value >= min && value <= max;
  
  if (!isValid) {
    input.classList.add('is-invalid');
    input.setCustomValidity(errorMessage);
  } else {
    input.classList.remove('is-invalid');
    input.setCustomValidity("");
  }
  
  // If the field is required and empty, show required validation
  if (input.required && input.value === '') {
    input.classList.add('is-invalid');
    input.setCustomValidity('This field is required');
  }
  
  input.reportValidity();
  return isValid;
}

// Make the validation functions globally available
window.validateInput = validateInput;
window.validateNumberInput = validateNumberInput;

// Function to reset the prediction form
function resetPredictionForm() {
  const form = document.getElementById('predictionForm');
  if (form) {
    form.reset();
    
    // Reset predict button state
    const predictButton = document.querySelector('button[type="submit"]');
    const predictButtonText = document.getElementById('predictButtonText');
    const predictButtonLoading = document.getElementById('predictButtonLoading');
    const predictButtonLoadingText = document.getElementById('predictButtonLoadingText');
    
    if (predictButton) predictButton.disabled = false;
    if (predictButtonText) predictButtonText.classList.remove('d-none');
    if (predictButtonLoading) predictButtonLoading.classList.add('d-none');
    if (predictButtonLoadingText) predictButtonLoadingText.classList.add('d-none');
    
    // Hide any previous results or error messages
    const resultDiv = document.getElementById('predictionResult');
    if (resultDiv) {
      resultDiv.style.display = 'none';
      resultDiv.innerHTML = '';
      resultDiv.className = 'mt-3 text-center fw-bold';
    }

    // Reset any custom validation messages
    const formInputs = form.querySelectorAll('input, select');
    formInputs.forEach(input => {
      if (input) {
        input.classList.remove('is-invalid');
        const feedback = input.nextElementSibling;
        if (feedback && feedback.classList && feedback.classList.contains('invalid-feedback')) {
          feedback.remove();
        }
      }
    });
    
    // Reset to first step in the stepper
    const firstStep = document.querySelector('.nav-link.active');
    if (firstStep) {
      firstStep.click();
    }
  }
}

// Initialize the application
async function initializeApp() {
  await loadValidOptions();
  
  // Add event listener to the form
  const form = document.getElementById("predictionForm");
  if (form) {
    form.addEventListener("submit", handlePredictionSubmit);
  }

  // Add input validation
  const countryInput = document.getElementById("COUNTRYNAME");
  const regionInput = document.getElementById("REGION");
  const startYearInput = document.getElementById("STARTYEAR");
  const startMonthInput = document.getElementById("STARTMONTH");
  const pptInput = document.getElementById("PPT");
  const tmaxInput = document.getElementById("TMAX");
  const soilMoistureInput = document.getElementById("SOILMOISTURE");

  if (countryInput) {
    countryInput.addEventListener("change", function () {
      const country = this.value.trim();
      if (country) {
        updateRegionListForCountry(country);
      }
    });
  }
  
  // Set current year as default for STARTYEAR
  if (startYearInput) {
    startYearInput.value = new Date().getFullYear();
  }
  
  // Set default month to current month
  if (startMonthInput) {
    startMonthInput.value = new Date().getMonth() + 1; // Months are 0-indexed in JS
  }

  // Add event listeners for country and region validation
  document.getElementById('COUNTRYNAME')?.addEventListener('change', function() {
    const country = this.value.trim();
    updateRegionListForCountry(country);
    
    // Clear region input when country changes
    const regionInput = document.getElementById('REGION');
    if (regionInput) {
      regionInput.value = '';
    }
  });
  
  // Add event listener for reset button
  document.getElementById('resetForm')?.addEventListener('click', function() {
    // Show confirmation dialog before resetting
    Swal.fire({
      title: 'Reset Form',
      text: 'Are you sure you want to reset the form? All entered data will be lost.',
      icon: 'warning',
      showCancelButton: true,
      confirmButtonColor: '#d33',
      cancelButtonColor: '#3085d6',
      confirmButtonText: 'Yes, reset it!',
      cancelButtonText: 'Cancel',
      allowOutsideClick: false
    }).then((result) => {
      if (result.isConfirmed) {
        // Show loading state on reset button
        const resetButton = document.getElementById('resetForm');
        const originalContent = resetButton.innerHTML;
        resetButton.disabled = true;
        resetButton.innerHTML = '<span class="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span>Resetting...';
        
        // Simulate a small delay for better UX
        // Show success toast first
        const Toast = Swal.mixin({
          toast: true,
          position: 'top-end',
          showConfirmButton: false,
          timer: 3000,
          timerProgressBar: true,
          didOpen: (toast) => {
            toast.addEventListener('mouseenter', Swal.stopTimer);
            toast.addEventListener('mouseleave', Swal.resumeTimer);
          },
          willClose: () => {
            // Only reset the form after toast is closed
            resetPredictionForm();
            // Reset button state
            resetButton.innerHTML = originalContent;
            resetButton.disabled = false;
          }
        });
        
        Toast.fire({
          icon: 'success',
          title: 'Form has been reset',
          background: 'var(--bs-success-bg-subtle)',
          color: 'var(--bs-success-text)',
          iconColor: 'var(--bs-success)'
        });
      }
    });
  });
}

// Initialize the app when DOM is loaded
document.addEventListener("DOMContentLoaded", initializeApp);

// Expose handlePredictionSubmit globally
window.handlePredictionSubmit = handlePredictionSubmit;

// Check if SweetAlert2 is loaded
if (typeof Swal === "undefined") {
  console.error(
    "SweetAlert2 failed to load. Please check your internet connection."
  );
}
