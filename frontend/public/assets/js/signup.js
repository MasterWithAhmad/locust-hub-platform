    async function handleSignup(event) {
      event.preventDefault();
      
      const nameInput = document.getElementById('name');
      const emailInput = document.getElementById('email');
      const passwordInput = document.getElementById('password');
      const securityQuestionInput = document.getElementById('securityQuestion');
      const securityAnswerInput = document.getElementById('securityAnswer');
      const errorDiv = document.getElementById('errorMessage');
      
      try {
        errorDiv.style.display = 'none';
        
        await api.auth.register(
          nameInput.value,
          emailInput.value,
          passwordInput.value,
          securityQuestionInput.value,
          securityAnswerInput.value
        );
        
        // If registration is successful, redirect to login page
        window.location.href = '/login.html?registered=true';
      } catch (error) {
        errorDiv.textContent = error.message;
        errorDiv.style.display = 'block';
      }
      
      return false;
    }