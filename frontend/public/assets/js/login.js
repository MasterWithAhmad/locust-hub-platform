document.addEventListener('DOMContentLoaded', function () {
      const urlParams = new URLSearchParams(window.location.search);
  
      // Show success message if account was deleted
      if (urlParams.has('accountDeleted')) {
        localStorage.removeItem('token');
        localStorage.removeItem('user');
        Swal.fire({
          icon: 'success',
          title: 'Account Deleted',
          text: 'Your account has been successfully deleted. We\'re sorry to see you go!',
          confirmButtonText: 'OK',
          allowOutsideClick: false,
          willClose: () => {
            window.history.replaceState({}, document.title, window.location.pathname);
          }
        });
      }
  
      // Check if user is already logged in
      const token = localStorage.getItem('token');
      if (token) {
        localStorage.removeItem('token');
        localStorage.removeItem('user');
        window.location.href = '/login.html';
        return;
      }
  
      // Attach form submit handler correctly
      const loginForm = document.getElementById('loginForm');
      if (loginForm) {
        loginForm.addEventListener('submit', handleLogin);
      }
  
      // Toggle password visibility
      const togglePassword = document.querySelector('#togglePassword');
      const password = document.querySelector('#password');
  
      if (togglePassword && password) {
        togglePassword.addEventListener('click', function () {
          const type = password.getAttribute('type') === 'password' ? 'text' : 'password';
          password.setAttribute('type', type);
  
          const icon = this.querySelector('i');
          if (icon) {
            icon.classList.toggle('bi-eye');
            icon.classList.toggle('bi-eye-slash');
          }
        });
      }
  
      // Show success message after registration
      if (urlParams.get('registered') === 'true') {
        const successDiv = document.getElementById('successMessage');
        if (successDiv) {
          successDiv.textContent = 'Registration successful! Please log in.';
          successDiv.style.display = 'block';
        }
        window.history.replaceState({}, document.title, window.location.pathname);
      }
    });
  
    async function handleLogin(event) {
      event.preventDefault();
  
      const email = document.getElementById('email').value;
      const password = document.getElementById('password').value;
      const rememberMe = document.getElementById('rememberMe').checked;
      const errorDiv = document.getElementById('errorMessage');
      const submitBtn = event.target.querySelector('button[type="submit"]');
      const originalBtnText = submitBtn.innerHTML;
  
      try {
        submitBtn.disabled = true;
        submitBtn.innerHTML = '<span class="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span>Logging in...';
        errorDiv.style.display = 'none';
  
        console.log('Attempting login with:', { email });
  
        const data = await window.api.auth.login(email, password, rememberMe);
  
        console.log('Login successful, token stored');
        window.location.href = 'dashboard.html';
  
      } catch (error) {
        console.error('Login error:', error);
        errorDiv.textContent = error.message || 'An error occurred during login. Please try again.';
        errorDiv.style.display = 'block';
        submitBtn.disabled = false;
        submitBtn.innerHTML = originalBtnText;
      }
      
    }