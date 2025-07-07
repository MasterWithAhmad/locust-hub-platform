document.addEventListener('DOMContentLoaded', function() {
    // Initialize Bootstrap tabs
    const tabPanes = document.querySelectorAll('.tab-pane');
    let currentTab = 0;
    const totalTabs = document.querySelectorAll('.nav-link[data-bs-toggle="pill"]').length;
    
    // Find the active tab on load
    const activeTab = document.querySelector('.nav-link.active');
    if (activeTab) {
        const tabId = activeTab.id;
        if (tabId && tabId.startsWith('step') && tabId.endsWith('-tab')) {
            currentTab = parseInt(tabId.replace('step', '').replace('-tab', '')) - 1;
        }
    }

    // Function to show a specific tab
    function showTab(index) {
        // Ensure index is within bounds
        if (index < 0) index = 0;
        if (index >= totalTabs) index = totalTabs - 1;
        
        // Update current tab
        currentTab = index;
        
        // Show the tab using Bootstrap's tab API
        const tabElement = document.querySelector(`#step${index + 1}-tab`);
        if (tabElement) {
            const tab = new bootstrap.Tab(tabElement);
            tab.show();
        }
        
        // Update navigation buttons
        updateButtons();
        
        // Update tab states
        updateTabStates();
    }
    
    // Function to update tab indicators (active/disabled states)
    function updateTabIndicators() {
        document.querySelectorAll('.nav-link[data-bs-toggle="pill"]').forEach((tab, index) => {
            const tabElement = document.querySelector(`#step${index + 1}-tab`);
            if (tabElement) {
                if (index <= currentTab) {
                    // Enable tabs up to current tab
                    tabElement.classList.remove('disabled');
                    tabElement.removeAttribute('disabled');
                } else {
                    // Disable tabs after current tab
                    tabElement.classList.add('disabled');
                    tabElement.setAttribute('disabled', 'disabled');
                }
                
                // Update active state
                if (index === currentTab) {
                    tabElement.classList.add('active');
                } else {
                    tabElement.classList.remove('active');
                }
            }
        });
    }
    
    // Function to validate current tab
    function validateTab(tabIndex) {
      const currentTab = document.querySelector(`#step${tabIndex + 1}`);
      if (!currentTab) {
        console.warn(`Tab ${tabIndex + 1} not found`);
        return false;
      }
      
      // Get all required inputs in the current tab
      const inputs = currentTab.querySelectorAll('input[required], select[required], textarea[required]');
      let valid = true;
      
      // Reset custom validation
      inputs.forEach(input => {
        input.classList.remove('is-invalid');
      });
      
      // Check each required field
      inputs.forEach(input => {
        if (!input.value.trim()) {
          input.classList.add('is-invalid');
          valid = false;
        }
      });
      
      // If any field is invalid, prevent form submission
      if (!valid) {
        const form = document.querySelector('form');
        if (form) {
          form.addEventListener('submit', function(e) {
            if (!valid) {
              e.preventDefault();
              e.stopPropagation();
            }
          }, { once: true });
        }
      }
      
      return valid;
      inputs.forEach(input => {
        if (!input.checkValidity()) {
          input.classList.add('is-invalid');
          valid = false;
        } else {
          input.classList.remove('is-invalid');
        }
      });
      return valid;
    }
    
    // Function to update navigation buttons
    function updateButtons() {
        // Update active tab indicator
        document.querySelectorAll('.nav-link').forEach((tab, index) => {
            if (index < currentTab) {
                tab.classList.add('completed');
            } else {
                tab.classList.remove('completed');
            }
        });
        
        // Update next/previous buttons visibility
        const prevButtons = document.querySelectorAll('.prev-step');
        const nextButtons = document.querySelectorAll('.next-step');
        
        prevButtons.forEach(btn => {
            btn.style.visibility = currentTab === 0 ? 'hidden' : 'visible';
        });
        
        nextButtons.forEach(btn => {
            if (currentTab === totalTabs - 1) {
                btn.innerHTML = '<i class="bi bi-lightning-charge-fill me-2"></i> Predict';
                btn.type = 'submit';
            } else {
                btn.innerHTML = 'Next <i class="bi bi-arrow-right ms-2"></i>';
                btn.type = 'button';
            }
        });
    }
    
    // Handle next/previous button clicks using event delegation
    document.addEventListener('click', function(e) {
        // Next button click
        const nextButton = e.target.closest('.next-step');
        if (nextButton) {
            e.preventDefault();
            e.stopPropagation();
            
            // Get the next step from data attribute or increment current tab
            const nextStep = nextButton.hasAttribute('data-next') 
                ? parseInt(nextButton.getAttribute('data-next')) 
                : currentTab + 2; // +2 because currentTab is 0-based and we want the next step
                
            if (validateTab(currentTab)) {
                showTab(nextStep - 1); // Convert to 0-based index
            } else {
                // Scroll to first invalid field if validation fails
                const currentPane = document.querySelector(`#step${currentTab + 1}`);
                const invalidField = currentPane.querySelector(':invalid');
                if (invalidField) {
                    invalidField.scrollIntoView({ behavior: 'smooth', block: 'center' });
                    invalidField.focus();
                }
            }
        }
        
        // Previous button click
        const prevButton = e.target.closest('.prev-step');
        if (prevButton) {
            e.preventDefault();
            e.stopPropagation();
            
            // Get the previous step from data attribute or decrement current tab
            const prevStep = prevButton.hasAttribute('data-prev') 
                ? parseInt(prevButton.getAttribute('data-prev')) 
                : currentTab; // currentTab is already 0-based
                
            showTab(prevStep - 1); // Convert to 0-based index
        }
    });
    
    // Make all tabs unclickable
    document.querySelectorAll('.nav-link[data-bs-toggle="pill"]').forEach(tab => {
        // Remove all click events
        tab.style.pointerEvents = 'none';
        tab.style.cursor = 'default';
        
        // Remove hover effects
        tab.style.opacity = '0.6';
    });
    
    // Make current tab active and clickable for visual feedback only
    function updateTabStates() {
        document.querySelectorAll('.nav-link[data-bs-toggle="pill"]').forEach((tab, index) => {
            if (index === currentTab) {
                tab.classList.add('active');
                tab.setAttribute('aria-selected', 'true');
            } else {
                tab.classList.remove('active');
                tab.setAttribute('aria-selected', 'false');
            }
        });
    }
    
    // Initialize
    updateButtons();
});
