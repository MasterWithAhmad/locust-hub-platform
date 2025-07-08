document.addEventListener('DOMContentLoaded', function () {
      // Check login status
      if (!api.auth.isLoggedIn()) {
        window.location.href = '/login.html';
        return;
      }

      // Show Bootstrap toast notification
      function showToast(type, message) {
        const toastContainer = document.getElementById('toastContainer');
        const toastId = 'toast-' + Date.now();
        const toast = document.createElement('div');
        toast.className = `toast align-items-center text-white bg-${type} border-0`;
        toast.role = 'alert';
        toast.setAttribute('aria-live', 'assertive');
        toast.setAttribute('aria-atomic', 'true');
        toast.id = toastId;

        toast.innerHTML = `
          <div class="d-flex">
            <div class="toast-body">
              ${message}
            </div>
            <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast" aria-label="Close"></button>
          </div>
        `;

        toastContainer.appendChild(toast);
        const bsToast = new bootstrap.Toast(toast, { autohide: true, delay: 5000 });
        bsToast.show();

        // Remove toast from DOM after it's hidden
        toast.addEventListener('hidden.bs.toast', function () {
          toast.remove();
        });
      }

      // Account Deletion Modal Logic
      const deleteModal = document.getElementById('deleteAccountModal');
      const confirmDeleteBtn = document.getElementById('confirmDeleteBtn');
      const confirmDeleteInput = document.getElementById('confirmDeleteInput');
      const confirmCheckbox = document.getElementById('confirmCheckbox');

      // Enable/disable delete button based on confirmation
      function updateDeleteButtonState() {
        const isConfirmed = confirmDeleteInput.value.toLowerCase() === 'delete my account';
        confirmDeleteBtn.disabled = !(isConfirmed && confirmCheckbox.checked);
      }

      confirmDeleteInput.addEventListener('input', updateDeleteButtonState);
      confirmCheckbox.addEventListener('change', updateDeleteButtonState);

      // Handle account deletion
      confirmDeleteBtn.addEventListener('click', async function () {
        if (!confirmDeleteBtn.disabled) {
          // Show SweetAlert confirmation
          const result = await Swal.fire({
            title: 'Are you absolutely sure?',
            text: 'This action cannot be undone. This will permanently delete your account and all associated data.',
            icon: 'warning',
            showCancelButton: true,
            confirmButtonColor: '#d33',
            cancelButtonColor: '#6c757d',
            confirmButtonText: 'Yes, delete my account',
            cancelButtonText: 'Cancel',
            reverseButtons: true,
            customClass: {
              confirmButton: 'btn btn-danger',
              cancelButton: 'btn btn-secondary me-2'
            },
            buttonsStyling: false
          });

          if (result.isConfirmed) {
            try {
              // Show loading state in SweetAlert
              Swal.fire({
                title: 'Deleting your account...',
                text: 'Please wait while we remove your data.',
                allowOutsideClick: false,
                didOpen: () => {
                  Swal.showLoading();
                }
              });

              // Call the API to delete the account
              const response = await api.user.deleteAccount();

              if (response.success) {
                // Show success message
                await Swal.fire({
                  icon: 'success',
                  title: 'Account Deleted',
                  text: 'Your account has been successfully deleted.',
                  confirmButtonText: 'Go to Login',
                  allowOutsideClick: false,
                  willClose: () => {
                    // Force a hard redirect to ensure user is logged out
                    window.location.href = response.redirect || '/login.html?accountDeleted=true';
                  }
                });

                // Force redirect after a short delay in case the user doesn't click the button
                setTimeout(() => {
                  window.location.href = response.redirect || '/login.html?accountDeleted=true';
                }, 2000);
              } else {
                throw new Error(response.message || 'Failed to delete account');
              }
            } catch (error) {
              console.error('Error deleting account:', error);

              // Hide loading state
              Swal.close();

              // Show error message
              await Swal.fire({
                icon: 'error',
                title: 'Deletion Failed',
                text: error.message || 'An error occurred while deleting your account. Please try again.',
                confirmButtonText: 'OK'
              });

              // Reset button state
              confirmDeleteBtn.disabled = false;
              confirmDeleteBtn.textContent = 'Delete My Account';
            }
          } else {
            // User cancelled the deletion
          }
        }
      });

      // Reset modal state when hidden
      const modalInstance = new bootstrap.Modal(deleteModal);
      deleteModal.addEventListener('hidden.bs.modal', function () {
        confirmDeleteInput.value = '';
        confirmCheckbox.checked = false;
        confirmDeleteBtn.disabled = true;
        confirmDeleteBtn.textContent = 'Delete My Account';
      });

      // Show toast if redirected after account deletion
      const urlParams = new URLSearchParams(window.location.search);
      if (urlParams.has('deleted')) {
        showToast('success', 'Your account has been successfully deleted.');
        // Clean up URL
        window.history.replaceState({}, document.title, window.location.pathname);
      }

      // Load user info if logged in (from localStorage initially)
      const user = api.auth.getCurrentUser();
      if (!user) {
        window.location.href = '/login.html';
        return;
      }

      // Update sidebar user profile (from localStorage initially)
      const initials = user.full_name.split(' ').map(n => n[0]).join('').toUpperCase();
      document.getElementById('userInitials').innerText = initials;
      document.getElementById('userName').innerText = user.full_name;

      // Populate profile information with initial data (from localStorage)
      document.getElementById('profileFullName').innerText = user.full_name || 'N/A';
      document.getElementById('profileEmail').innerText = user.email || 'N/A';
      document.getElementById('profileCreatedAt').innerText = 'Loading...'; // Placeholder

      // Fetch full user details from backend to get created_at and potentially update other info
      async function fetchUserProfile() {
        try {
          // Use the existing API function to fetch authenticated user profile
          const response = await api.user.getUserDetails();

          // Check if the response has status 'success' and contains data
          if (response && response.status === 'success' && response.data) {
            const fullUser = response.data;

            // Populate profile information from backend data
            document.getElementById('profileFullName').innerText = fullUser.full_name || 'N/A';
            document.getElementById('profileEmail').innerText = fullUser.email || 'N/A';

            // Format and display the created_at date if available
            if (fullUser.created_at) {
              // Handle potential different date formats (ISO string or Date object)
              const dateValue = typeof fullUser.created_at === 'string'
                ? new Date(fullUser.created_at)
                : (fullUser.created_at instanceof Date ? fullUser.created_at : null);

              if (dateValue && !isNaN(dateValue)) {
                const options = { year: 'numeric', month: 'long', day: 'numeric' };
                document.getElementById('profileCreatedAt').innerText = dateValue.toLocaleDateString(undefined, options);
              } else {
                console.error('Invalid created_at date format:', fullUser.created_at);
                document.getElementById('profileCreatedAt').innerText = 'Invalid date format';
              }
            } else {
              document.getElementById('profileCreatedAt').innerText = 'Date not available';
            }

            // Optionally update localStorage with full user data if needed for other pages
            // localStorage.setItem('user', JSON.stringify(fullUser));

          } else {
            console.error('Failed to fetch full user profile: Invalid response format or status', response);
            // Keep initial data or set to error
            document.getElementById('profileCreatedAt').innerText = response?.message || 'Error loading date';
          }
        } catch (error) {
          console.error('Error fetching user profile:', error);
          document.getElementById('profileCreatedAt').innerText = 'Error loading date';
        }
      }

      // Load current user data into the form
      function loadCurrentUserData() {
        const currentUser = api.auth.getCurrentUser();
        if (currentUser) {
          const fullNameInput = document.getElementById('updateFullName');
          const emailInput = document.getElementById('updateEmail');

          if (fullNameInput) fullNameInput.value = currentUser.full_name || '';
          if (emailInput) emailInput.value = currentUser.email || '';

          console.log('Loaded user data:', {
            fullName: currentUser.full_name,
            email: currentUser.email
          });
        } else {
          console.error('No current user found');
        }
      }

      // Initialize user profile and load data into forms
      console.log('DOM fully loaded, initializing profile and forms...');
      fetchUserProfile();
      loadCurrentUserData();

      // Set up form submission handler for profile update
      const updateProfileForm = document.getElementById('updateProfileForm');
      if (updateProfileForm) {
        console.log('Found update profile form, adding submit handler...');

        const submitButton = document.getElementById('updateProfileBtn');

        async function handleProfileUpdate(event) {
          console.log('Form/button event triggered for profile update');
          if (event) {
            event.preventDefault();
            event.stopPropagation();
          }

          const updateFullNameInput = document.getElementById('updateFullName');
          const updateEmailInput = document.getElementById('updateEmail');

          if (!updateFullNameInput || !updateEmailInput || !submitButton) {
            console.error('Could not find all required form elements for profile update');
            return;
          }

          const full_name = updateFullNameInput.value.trim();
          const email = updateEmailInput.value.trim();

          console.log('Profile update form values:', { full_name, email });

          const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
          if (!emailRegex.test(email)) {
            await Swal.fire({
              icon: 'error',
              title: 'Invalid Email',
              text: 'Please enter a valid email address',
              confirmButtonColor: '#3085d6',
            });
            return;
          }

          if (!full_name || !email) {
            await Swal.fire({
              icon: 'error',
              title: 'Error',
              text: 'Please fill in all required fields',
              confirmButtonColor: '#3085d6',
            });
            return;
          }

          const originalButtonText = submitButton.innerHTML;
          submitButton.disabled = true;
          submitButton.innerHTML = '<span class="spinner-border spinner-border-sm me-1" role="status" aria-hidden="true"></span> Updating...';

          try {
            const response = await fetch('/api/user/profile', {
              method: 'PUT',
              headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${localStorage.getItem('token')}`
              },
              body: JSON.stringify({
                full_name: full_name,
                email: email
              })
            });

            const data = await response.json();

            if (!response.ok) {
              throw new Error(data.error || 'Failed to update profile');
            }

            await Swal.fire({
              icon: 'success',
              title: 'Success!',
              text: 'Your profile has been updated successfully.',
              confirmButtonColor: '#3085d6',
              confirmButtonText: 'OK'
            });

            const profileFullName = document.getElementById('profileFullName');
            const profileEmail = document.getElementById('profileEmail');

            if (profileFullName) profileFullName.textContent = full_name;
            if (profileEmail) profileEmail.textContent = email;

            const userData = JSON.parse(localStorage.getItem('user') || '{}');
            userData.full_name = full_name;
            userData.email = email;
            localStorage.setItem('user', JSON.stringify(userData));

            if (window.auth && typeof window.auth.setUser === 'function') {
              window.auth.setUser(userData);
            }

            const initialsElement = document.getElementById('userInitials');
            const userNameElement = document.getElementById('userName');
            if (initialsElement) {
              const initials = full_name.split(' ').map(n => n[0]).join('').toUpperCase();
              initialsElement.textContent = initials;
            }
            if (userNameElement) {
              userNameElement.textContent = full_name;
            }

          } catch (error) {
            console.error('Error updating profile:', error);
            let errorMessage = 'Failed to update profile. Please try again.';
            let errorTitle = 'Update Failed';
            let showReload = false;

            if (error.message) {
              if (error.message.includes('Failed to fetch')) {
                errorMessage = 'Unable to connect to the server. Please check your internet connection and try again.';
                showReload = true;
              } else if (error.message.includes('email already in use')) {
                errorMessage = 'This email address is already in use. Please use a different email.';
              } else if (error.message.includes('invalid token') || error.message.includes('jwt expired')) {
                errorMessage = 'Your session has expired. Please log in again.';
                showReload = true;
                setTimeout(() => {
                  window.location.href = '/login.html';
                }, 2000);
              } else {
                errorMessage = error.message;
              }
            }

            const result = await Swal.fire({
              icon: 'error',
              title: errorTitle,
              text: errorMessage,
              confirmButtonColor: '#3085d6',
              confirmButtonText: showReload ? 'Reload Page' : 'OK',
              showCancelButton: showReload,
              cancelButtonText: 'Cancel',
              allowOutsideClick: !showReload
            });

            if (showReload && result.isConfirmed) {
              window.location.reload();
            }
          } finally {
            if (submitButton) {
              submitButton.disabled = false;
              submitButton.innerHTML = originalButtonText;
            }
          }
        }

        if (submitButton) {
          submitButton.addEventListener('click', handleProfileUpdate);
        }
        updateProfileForm.addEventListener('submit', handleProfileUpdate);
      } else {
        console.error('Update profile form not found.');
      }

      // Factory Reset Modal Logic
      const factoryResetModal = document.getElementById('factoryResetModal');
      const confirmFactoryResetBtn = document.getElementById('confirmFactoryResetBtn');
      const confirmFactoryResetInput = document.getElementById('confirmFactoryResetInput');

      function updateFactoryResetButtonState() {
        const isConfirmed = confirmFactoryResetInput.value.toLowerCase() === 'reset my data';
        confirmFactoryResetBtn.disabled = !isConfirmed;
      }

      if (confirmFactoryResetInput) {
        confirmFactoryResetInput.addEventListener('input', updateFactoryResetButtonState);
      }

      if (confirmFactoryResetBtn) {
        confirmFactoryResetBtn.addEventListener('click', async function () {
          if (!confirmFactoryResetBtn.disabled) {
            // Show SweetAlert confirmation
            const result = await Swal.fire({
              title: 'Are you sure?',
              text: 'This will delete all your predictions and cannot be undone. Your account will remain.',
              icon: 'warning',
              showCancelButton: true,
              confirmButtonColor: '#f6c23e',
              cancelButtonColor: '#6c757d',
              confirmButtonText: 'Yes, reset my data',
              cancelButtonText: 'Cancel',
              reverseButtons: true,
              customClass: {
                confirmButton: 'btn btn-warning',
                cancelButton: 'btn btn-secondary me-2'
              },
              buttonsStyling: false
            });

            if (result.isConfirmed) {
              try {
                Swal.fire({
                  title: 'Resetting your data...',
                  text: 'Please wait while we clear your predictions.',
                  allowOutsideClick: false,
                  didOpen: () => {
                    Swal.showLoading();
                  }
                });

                // Call the API to reset user data
                const response = await fetch('/api/user/factory-reset', {
                  method: 'POST',
                  headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${localStorage.getItem('token')}`
                  }
                });

                const data = await response.json();

                if (data.success) {
                  await Swal.fire({
                    icon: 'success',
                    title: 'Factory Reset Complete',
                    text: data.message || 'All your predictions have been deleted.',
                    confirmButtonText: 'OK'
                  });
                  // Optionally, refresh the page or update UI
                } else {
                  throw new Error(data.message || 'Failed to reset data');
                }
              } catch (error) {
                Swal.close();
                await Swal.fire({
                  icon: 'error',
                  title: 'Reset Failed',
                  text: error.message || 'An error occurred while resetting your data. Please try again.',
                  confirmButtonText: 'OK'
                });
              }
            }
          }
        });
      }

      if (factoryResetModal) {
        factoryResetModal.addEventListener('hidden.bs.modal', function () {
          if (confirmFactoryResetInput) confirmFactoryResetInput.value = '';
          if (confirmFactoryResetBtn) confirmFactoryResetBtn.disabled = true;
        });
      }
    }); // End of DOMContentLoaded

    // Password Change Handler
    function setupPasswordChange() {
      const form = document.getElementById('changePasswordForm');
      if (!form) {
        console.error('Change password form not found.');
        return;
      }

      form.addEventListener('submit', handlePasswordSubmit);
    }

    async function handlePasswordSubmit(event) {
      event.preventDefault();
      console.log('Form/button event triggered for password change');

      const form = event.target;
      const currentPasswordInput = form.currentPassword;
      const newPasswordInput = form.newPassword;
      const confirmNewPasswordInput = form.confirmPassword;

      const currentPassword = currentPasswordInput?.value;
      const newPassword = newPasswordInput?.value;
      const confirmPassword = confirmNewPasswordInput?.value;
      const submitButton = form.querySelector('button[type="submit"]');

      if (!currentPasswordInput || !newPasswordInput || !confirmNewPasswordInput || !submitButton) {
        console.error('Could not find all required form elements for password change');
        return;
      }

      if (newPassword !== confirmPassword) {
        showError('New password and confirm password do not match.');
        return;
      }

      if (newPassword.length < 8) {
        showError('Password must be at least 8 characters long.');
        return;
      }

      const originalButtonHTML = submitButton.innerHTML;
      submitButton.disabled = true;
      submitButton.innerHTML = '<span class="spinner-border spinner-border-sm me-1" role="status" aria-hidden="true"></span>Updating...';

      try {
        // Use the API method from api.js
        const response = await api.user.changePassword({
          current_password: currentPassword,
          new_password: newPassword
        });

        console.log('Change password response:', response);
        form.reset(); // Clear form on success
        showSuccess(response.message || 'Password changed successfully!');

      } catch (error) {
        console.error('Password change error:', error);
        const errorMessage = error.message === 'Failed to change password' && error.response?.data?.message 
          ? error.response.data.message 
          : error.message || 'Failed to change password. Please try again.';
        showError(errorMessage);
      } finally {
        submitButton.disabled = false;
        submitButton.innerHTML = originalButtonHTML; // Restore original button HTML
      }
    }

    function showError(message) {
      Swal.fire({
        icon: 'error',
        title: 'Error',
        text: message,
        confirmButtonColor: '#3085d6'
      });
    }

    function showSuccess(message) {
      Swal.fire({
        icon: 'success',
        title: 'Success!',
        text: message,
        confirmButtonColor: '#3085d6'
      });
    }

    // Initialize when DOM is loaded
    if (document.readyState === 'loading') {
      document.addEventListener('DOMContentLoaded', setupPasswordChange);
    } else {
      setupPasswordChange(); // Call it directly if DOM is already loaded
    }