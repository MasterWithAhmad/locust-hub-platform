// Global configuration
const CONFIG = {
  API_BASE_URL: 'http://127.0.0.1:5000',
  ENDPOINTS: {
    FORGOT_PASSWORD: '/api/auth/forgot-password',
    VERIFY_ANSWER: '/api/auth/verify-answer',
    RESET_PASSWORD: '/api/auth/reset-password'
  }
};

// Global state
const state = {
  currentStep: 1,
  userEmail: '',
  securityQuestion: ''
};

// Initialize SweetAlert2 with theme
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
});

// Password strength checker
function checkPasswordStrength(password) {
  let strength = 0;
  const tips = [];
  
  if (!password) {
    return { strength: 0, tips: ['Enter a password'] };
  }
  
  if (password.length < 8) {
    tips.push('at least 8 characters');
  } else {
    strength += 1;
  }
  
  if (password.match(/[a-z]+/)) {
    strength += 1;
  } else {
    tips.push('lowercase letter');
  }
  
  if (password.match(/[A-Z]+/)) {
    strength += 1;
  } else {
    tips.push('uppercase letter');
  }
  
  if (password.match(/[0-9]+/)) {
    strength += 1;
  } else {
    tips.push('number');
  }
  
  if (password.match(/[!@#$%^&*(),.?":{}|<>]+/)) {
    strength += 1;
  } else {
    tips.push('special character');
  }
  
  // Only update UI if elements exist
  if (progressBar && passwordStrength && passwordStrengthText) {
    const width = (strength / 5) * 100;
    progressBar.style.width = `${width}%`;
    
    // Update strength text and color
    let strengthText = '';
    let strengthClass = '';
    
    if (strength <= 1) {
      strengthText = 'Very Weak';
      strengthClass = 'bg-danger';
    } else if (strength <= 2) {
      strengthText = 'Weak';
      strengthClass = 'bg-warning';
    } else if (strength <= 3) {
      strengthText = 'Good';
      strengthClass = 'bg-info';
    } else if (strength <= 4) {
      strengthText = 'Strong';
      strengthClass = 'bg-primary';
    } else {
      strengthText = 'Very Strong';
      strengthClass = 'bg-success';
    }
    
    passwordStrength.className = 'progress-bar ' + strengthClass;
    passwordStrengthText.textContent = strengthText;
  }
  
  return {
    strength: strength,
    tips: tips
  };
}

// Initialize when DOM is loaded
document.addEventListener("DOMContentLoaded", function () {
  // Initialize tooltips
  const tooltipTriggerList = [].slice.call(
    document.querySelectorAll('[data-bs-toggle="tooltip"]')
  );
  tooltipTriggerList.map(function (tooltipTriggerEl) {
    return new bootstrap.Tooltip(tooltipTriggerEl);
  });

  // Password visibility toggles
  const toggleNewPassword = document.getElementById('toggleNewPassword');
  const toggleConfirmPassword = document.getElementById('toggleConfirmPassword');

  // Toggle new password visibility
  if (toggleNewPassword) {
    toggleNewPassword.addEventListener('click', function() {
      const input = document.getElementById('newPassword');
      const icon = this.querySelector('i');
      if (input && icon) {
        if (input.type === 'password') {
          input.type = 'text';
          icon.classList.remove('bi-eye');
          icon.classList.add('bi-eye-slash');
        } else {
          input.type = 'password';
          icon.classList.remove('bi-eye-slash');
          icon.classList.add('bi-eye');
        }
      }
    });
  }

  // Toggle confirm password visibility
  if (toggleConfirmPassword) {
    toggleConfirmPassword.addEventListener('click', function() {
      const input = document.getElementById('confirmPassword');
      const icon = this.querySelector('i');
      if (input && icon) {
        if (input.type === 'password') {
          input.type = 'text';
          icon.classList.remove('bi-eye');
          icon.classList.add('bi-eye-slash');
        } else {
          input.type = 'password';
          icon.classList.remove('bi-eye-slash');
          icon.classList.add('bi-eye');
        }
      }
    });
  }
  
  // Password strength checker
  if (newPasswordInput) {
    newPasswordInput.addEventListener('input', function() {
      checkPasswordStrength(this.value);
      validatePasswords();
    });
  }
  
  if (confirmPasswordInput) {
    confirmPasswordInput.addEventListener('input', validatePasswords);
  }

  // Initialize form state
  showStep(1);
  updateProgressBar();

  // Expose functions to global scope
  window.checkEmail = checkEmail;
  window.verifyAnswer = verifyAnswer;
  window.resetPassword = resetPassword;
  window.backToStep = backToStep;
  window.backToStep1 = () => backToStep(1);
  window.backToStep2 = () => backToStep(2);
});

// Verify security answer
async function verifySecurityAnswer() {
    const answer = document.getElementById('securityAnswer');
    const submitButton = document.getElementById('verifyAnswerBtn');
    
    if (!answer || !answer.value.trim()) {
        Toast.fire({ icon: 'error', title: 'Please enter your security answer' });
        return;
    }
    
    try {
        // Show loading state
        const originalText = submitButton.innerHTML;
        submitButton.disabled = true;
        submitButton.innerHTML = '<span class="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span>Verifying...';
        
        const response = await fetch(`${API_BASE_URL}/api/auth/verify-answer`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                email: userEmail,
                answer: answer.value.trim()
            })
        });
        
        if (!response.ok) {
            const error = await response.json().catch(() => ({}));
            throw new Error(error.message || 'Invalid security answer');
        }
        
        Toast.fire({ icon: 'success', title: 'Security answer verified' });
        showStep(3);
    } catch (error) {
        console.error('Error verifying security answer:', error);
        Toast.fire({ 
            icon: 'error', 
            title: error.message || 'Failed to verify security answer. Please try again.' 
        });
    } finally {
        if (submitButton) {
            submitButton.disabled = false;
            submitButton.innerHTML = 'Verify Answer <i class="bi bi-arrow-right ms-2"></i>';
        }
    }
}

// Reset password
async function resetPassword() {
    const newPassword = document.getElementById('newPassword');
    const confirmPassword = document.getElementById('confirmPassword');
    const submitButton = document.getElementById('resetPasswordBtn');
    
    if (!newPassword || !confirmPassword) {
        Toast.fire({ icon: 'error', title: 'Password fields not found' });
        return;
    }
    
    const newPasswordValue = newPassword.value.trim();
    const confirmPasswordValue = confirmPassword.value.trim();
    
    if (!newPasswordValue || !confirmPasswordValue) {
        Toast.fire({ icon: 'error', title: 'Please enter and confirm your new password' });
        return;
    }
    
    if (newPasswordValue !== confirmPasswordValue) {
        Toast.fire({ icon: 'error', title: 'Passwords do not match' });
        return;
    }
    
    try {
        // Show loading state
        const originalText = submitButton.innerHTML;
        submitButton.disabled = true;
        submitButton.innerHTML = '<span class="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span>Resetting...';
        
        const response = await fetch(`${API_BASE_URL}/auth/reset-password`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                email: userEmail,
                new_password: newPasswordValue
            })
        });
        
        if (!response.ok) {
            const error = await response.json().catch(() => ({}));
            throw new Error(error.message || 'Failed to reset password');
        }
        
        Toast.fire({ 
            icon: 'success', 
            title: 'Password reset successfully!',
            showConfirmButton: false,
            timer: 2000
        });
        
        // Redirect to login after a short delay
        setTimeout(() => {
            window.location.href = 'login.html';
        }, 2000);
        
    } catch (error) {
        console.error('Error resetting password:', error);
        Toast.fire({ 
            icon: 'error', 
            title: error.message || 'Failed to reset password. Please try again.' 
        });
    } finally {
        if (submitButton) {
            submitButton.disabled = false;
            submitButton.innerHTML = 'Reset Password';
        }
    }
}

// Check email and proceed to next step
async function checkEmail(button) {
    console.log('checkEmail function called');
    
    const email = document.getElementById('email');
    const submitButton = document.getElementById('emailVerifyBtn');
    
    if (!email) {
        Toast.fire({ icon: 'error', title: 'Email input element not found' });
        return;
    }
    
    const emailValue = email.value.trim();
    if (!emailValue) {
        Toast.fire({ icon: 'error', title: 'Please enter your email address' });
        return;
    }
    
    try {
        // Show loading state
        const originalText = submitButton.innerHTML;
        submitButton.disabled = true;
        submitButton.innerHTML = '<span class="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span>Verifying...';
        
        // Call API to verify email
        const response = await fetch(`${API_BASE_URL}/api/auth/verify-answer`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ email: emailValue })
        });

        if (!response.ok) {
            const error = await response.json().catch(() => ({}));
            throw new Error(error.message || 'Failed to verify email');
        }

        const data = await response.json();
        userEmail = emailValue;
        securityQuestion = data.security_question;
        
        // Update security question text
        const questionElement = document.getElementById('securityQuestionText');
        if (questionElement) {
            questionElement.textContent = securityQuestion;
        }
        
        Toast.fire({ icon: 'success', title: 'Email verified successfully' });
        showStep(2);
    } catch (error) {
        console.error('Error in checkEmail:', error);
        Toast.fire({ 
            icon: 'error', 
            title: error.message || 'Failed to verify email. Please try again.' 
        });
    } finally {
        if (submitButton) {
            submitButton.disabled = false;
            submitButton.innerHTML = 'Continue <i class="bi bi-arrow-right ms-2"></i>';
        }
    }
}

// Show/hide steps
function showStep(stepNumber) {
    // Hide all steps first
    document.querySelectorAll('.step').forEach(step => {
        step.style.display = 'none';
    });
    
    // Show the current step
    const currentStep = document.getElementById(`step${stepNumber}`);
    if (currentStep) {
        currentStep.style.display = 'block';
    }
    
    // Update progress indicator
    updateProgress(stepNumber);
}

// Update progress indicator
function updateProgress(stepNumber) {
    const progress = document.querySelector('.progress-bar');
    if (progress) {
        const percentage = ((stepNumber - 1) / 2) * 100;
        progress.style.width = `${percentage}%`;
        progress.setAttribute('aria-valuenow', percentage);
    }
}

// Validate password match
function validatePasswords() {
  const password = newPasswordInput?.value || '';
  const confirm = confirmPasswordInput?.value || '';
  const resetBtn = document.getElementById('resetPasswordBtn');
  
  if (!resetBtn) return;
  
  if (password && confirm) {
    if (password !== confirm) {
      resetBtn.disabled = true;
      resetBtn.title = 'Passwords do not match';
      return false;
    } else {
      const strength = checkPasswordStrength(password);
      resetBtn.disabled = strength.strength < 3; // Require at least 'Good' strength
      resetBtn.title = strength.strength < 3 ? 'Password is too weak' : '';
      return strength.strength >= 3;
    }
  }
  
function backToStep(step) {
  // If going back from step 3 to step 2, clear the password fields
  if (currentStep === 3 && step === 2) {
    document.getElementById('newPassword').value = '';
    document.getElementById('confirmPassword').value = '';
    
    // Reset password strength meter
    if (passwordStrength) {
      passwordStrength.style.width = '0%';
      passwordStrength.className = 'progress-bar';
    }
    
    if (passwordStrengthText) {
      passwordStrengthText.textContent = 'Very Weak';
    }
  }
  
  showStep(step);
}

// Show error message with better formatting
function showError(message, targetId = null) {
  // If targetId is provided, show error in that element
  if (targetId) {
    const target = document.getElementById(targetId);
    if (target) {
      // If target is a form text element, update it
      if (target.classList.contains('form-text')) {
        target.classList.remove('text-muted');
        target.classList.add('text-danger');
        target.textContent = message;
        return;
      }
      
      // Otherwise, create or update error message below the target
      let errorElement = target.nextElementSibling;
      if (!errorElement || !errorElement.classList.contains('invalid-feedback')) {
        errorElement = document.createElement('div');
        errorElement.className = 'invalid-feedback d-block';
        target.parentNode.insertBefore(errorElement, target.nextSibling);
      }
      errorElement.textContent = message;
      
      // Add is-invalid class to input
      if (target.tagName === 'INPUT' || target.tagName === 'SELECT' || target.tagName === 'TEXTAREA') {
        target.classList.add('is-invalid');
      }
      
      // Scroll to the error
      target.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
  } else {
    // Fallback to toast notification
    Toast.fire({
      icon: 'error',
      title: 'Error',
      text: message
    });
  }
}

// Clear error message
function clearError(targetId) {
  if (targetId) {
    const target = document.getElementById(targetId);
    if (target) {
      // If target is a form text element, reset it
      if (target.classList.contains('form-text')) {
        target.classList.remove('text-danger');
        target.classList.add('text-muted');
        target.textContent = target.dataset.originalText || '';
        return;
      }
      
      // Remove is-invalid class from input
      target.classList.remove('is-invalid');
      
      // Remove error message element if it exists
      const errorElement = target.nextElementSibling;
      if (errorElement && errorElement.classList.contains('invalid-feedback')) {
        errorElement.remove();
      }
    }
  }
}

// Update progress bar based on current step
function updateProgressBar(step) {
  const progressBar = document.getElementById('progress-bar');
  if (progressBar) {
    const progress = ((step - 1) / 2) * 100;
    progressBar.style.width = `${progress}%`;
  }
}

// Show a specific step in the form with smooth transition
function showStep(step) {
  // Update state
  state.currentStep = step;
  
  // Hide all steps
  document.querySelectorAll('.step-content').forEach(el => {
    el.style.display = 'none';
  });
  
  // Show the current step
  const currentStepEl = document.getElementById(`step${step}`);
  if (currentStepEl) {
    currentStepEl.style.display = 'block';
  }
  
  // Update step indicators
  document.querySelectorAll('.step').forEach((el, index) => {
    const stepNumber = index + 1;
    if (stepNumber < step) {
      el.classList.add('completed');
      el.classList.remove('active');
    } else if (stepNumber === step) {
      el.classList.add('active');
      el.classList.remove('completed');
    } else {
      el.classList.remove('active', 'completed');
    }
  });
  
  // Update progress bar
  const progressBar = document.getElementById('progress-bar');
  if (progressBar) {
    const progress = ((step - 1) / 2) * 100;
    progressBar.style.width = `${progress}%`;
  }
  
  // Focus on first input of the current step
  const firstInput = currentStepEl?.querySelector('input');
  if (firstInput) {
    firstInput.focus();
  }
}

// Validate password match
function validatePasswords() {
  const newPassword = document.getElementById('newPassword')?.value;
  const confirmPassword = document.getElementById('confirmPassword')?.value;
  const errorElement = document.getElementById('password-match-error');
  
  if (!newPassword || !confirmPassword) {
    if (errorElement) errorElement.textContent = '';
    return false;
  }
  
  if (newPassword !== confirmPassword) {
    if (errorElement) {
      errorElement.textContent = 'Passwords do not match';
      errorElement.style.display = 'block';
    }
    return false;
  }
  
  if (errorElement) {
    errorElement.textContent = '';
    errorElement.style.display = 'none';
  }
  return true;
}

// Reset password function
async function resetPassword() {
  const newPassword = document.getElementById('newPassword')?.value.trim();
  const confirmPassword = document.getElementById('confirmPassword')?.value.trim();
  const resetBtn = document.querySelector('button[onclick="resetPassword()"]');
  
  // Basic validation
  if (!newPassword || !confirmPassword) {
    showError('Please fill in all password fields', 'passwordHelp');
    return;
  }
  
  if (newPassword !== confirmPassword) {
    showError('Passwords do not match', 'passwordHelp');
    return;
  }
  
  // Check password strength
  const strength = checkPasswordStrength(newPassword);
  if (strength.strength < 3) {
    showError('Password is too weak. Please choose a stronger password.', 'passwordHelp');
    return;
  }
  
  // Store button state for restoration
  let buttonState = {
    html: resetBtn?.innerHTML || 'Reset Password',
    disabled: resetBtn?.disabled || false
  };
  
  try {
    // Show loading state
    if (resetBtn) {
      resetBtn.disabled = true;
      resetBtn.innerHTML = '<span class="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span> Resetting...';
    }
    
    // Call API to reset password
    const response = await fetch(`${API_BASE_URL}/api/auth/reset-password`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        email: userEmail,
        new_password: newPassword,
        reset_token: localStorage.getItem('resetToken')
      })
    });
    
    const data = await response.json();
    
    if (!response.ok) {
      throw new Error(data.message || 'Failed to reset password. Please try again.');
    }
    
    // Show success message
    await Swal.fire({
      icon: 'success',
      title: 'Password Reset Successful!',
      text: 'Your password has been updated successfully. You can now log in with your new password.',
      confirmButtonText: 'Go to Login'
    });
    
    // Clear stored data and redirect to login
    localStorage.removeItem('resetToken');
    window.location.href = 'login.html';
    
  } catch (error) {
    console.error('Error resetting password:', error);
    showError(error.message || 'An error occurred while resetting your password. Please try again.', 'passwordHelp');
    
    // Re-throw the error to be caught by the global error handler if needed
    throw error;
  } finally {
    // Reset button state
    if (resetBtn) {
      resetBtn.disabled = buttonState.disabled;
      resetBtn.innerHTML = buttonState.html;
    }
  }
}

// Validate email format
function isValidEmail(email) {
  const re = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  return re.test(String(email).toLowerCase());
}

// Toggle password visibility
function togglePassword(inputId, button) {
  const input = document.getElementById(inputId);
  if (!input) return;
  
  const icon = button.querySelector('i');
  
  if (input.type === 'password') {
    input.type = 'text';
    icon.classList.remove('bi-eye');
    icon.classList.add('bi-eye-slash');
    button.setAttribute('aria-label', 'Hide password');
  } else {
    input.type = 'password';
    icon.classList.remove('bi-eye-slash');
    icon.classList.add('bi-eye');
    button.setAttribute('aria-label', 'Show password');
  }
  
  // Focus back on the input
  input.focus();
}

// Show error message
function showError(message, targetId = null) {
  let errorElement;
  
  if (targetId) {
    errorElement = document.getElementById(targetId);
    if (!errorElement) {
      // If target ID not found, try to find a help element
      const input = document.querySelector(`[aria-describedby="${targetId}"]`);
      if (input) {
        errorElement = document.getElementById(targetId);
      }
    }
  } else {
    errorElement = document.getElementById('error-message');
  }
  
  if (errorElement) {
    errorElement.textContent = message;
    errorElement.style.display = 'block';
    errorElement.classList.add('text-danger');
    
    // Auto-hide after 5 seconds
    setTimeout(() => {
      errorElement.style.display = 'none';
    }, 5000);
  } else {
    console.error('Error element not found');
    Toast.fire({
      icon: 'error',
      title: 'Error',
      text: message
    });
  }
}

// Clear error message
function clearError(targetId) {
  const errorElement = document.getElementById(targetId);
  if (errorElement) {
    errorElement.textContent = '';
    errorElement.style.display = 'none';
    errorElement.classList.remove('text-danger');
  }
}

// Toggle password visibility
function togglePassword(inputId, button) {
  const input = document.getElementById(inputId);
  if (!input) return;
  
  const icon = button.querySelector('i');
  
  if (input.type === 'password') {
    input.type = 'text';
    icon.classList.remove('bi-eye');
    icon.classList.add('bi-eye-slash');
  } else {
    input.type = 'password';
    icon.classList.remove('bi-eye-slash');
    icon.classList.add('bi-eye');
  }
  
  // Focus the input after toggling
  input.focus();
}

// Show a specific step in the form
function showStep(step) {
  // Update state
  state.currentStep = step;
  
  // Hide all steps
  document.querySelectorAll('.step-content').forEach(el => {
    el.style.display = 'none';
  });
  
  // Show the current step
  const currentStepEl = document.getElementById(`step${step}`);
  if (currentStepEl) {
    currentStepEl.style.display = 'block';
  }
  
  // Update step indicators
  document.querySelectorAll('.step').forEach((el, index) => {
    const stepNumber = index + 1;
    if (stepNumber < step) {
      el.classList.add('completed');
      el.classList.remove('active');
    } else if (stepNumber === step) {
      el.classList.add('active');
      el.classList.remove('completed');
    } else {
      el.classList.remove('active', 'completed');
    }
  });
  
  // Update progress bar
  const progressBar = document.getElementById('progress-bar');
  if (progressBar) {
    const progress = ((step - 1) / 2) * 100;
    progressBar.style.width = `${progress}%`;
  }
  
  // Focus on first input of the current step
  const firstInput = currentStepEl?.querySelector('input');
  if (firstInput) {
    firstInput.focus();
  }
}

// Navigate back to a specific step
function backToStep(step) {
  // If going back from step 3 to step 2, clear the password fields
  if (state.currentStep === 3 && step === 2) {
    document.getElementById('newPassword').value = '';
    document.getElementById('confirmPassword').value = '';
    const strengthMeter = document.getElementById('password-strength');
    if (strengthMeter) strengthMeter.style.width = '0%';
  }
  
  showStep(step);
}

// Check email and get security question
async function checkEmail() {
  const emailInput = document.getElementById('email');
  const email = emailInput.value.trim();
  const button = document.querySelector('#step1 button[type="submit"]');
  const originalText = button.innerHTML;
  
  // Validate email
  if (!email) {
    showError('Please enter your email address', 'emailHelp');
    emailInput.focus();
    return;
  }
  
  if (!isValidEmail(email)) {
    showError('Please enter a valid email address', 'emailHelp');
    emailInput.focus();
    return;
  }
  
  try {
    // Show loading state
    button.disabled = true;
    button.innerHTML = '<span class="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span> Sending...';
    
    // Call API to check email
    const response = await fetch(`${CONFIG.API_BASE_URL}${CONFIG.ENDPOINTS.FORGOT_PASSWORD}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email })
    });
    
    const data = await response.json();
    
    if (!response.ok) {
      throw new Error(data.message || 'Failed to send verification code');
    }
    
    // Store user email and security question
    state.userEmail = email;
    state.securityQuestion = data.security_question || 'What is your mother\'s maiden name?';
    
    // Update UI
    document.getElementById('securityQuestionText').textContent = state.securityQuestion;
    showStep(2);
    
    // Show success message
    Toast.fire({
      icon: 'success',
      title: 'Verification sent!',
      text: 'Please check your email for the verification code.'
    });
    
  } catch (error) {
    console.error('Error:', error);
    showError(error.message || 'Failed to send verification code', 'emailHelp');
  } finally {
    // Reset button state
    button.disabled = false;
    button.innerHTML = originalText;
  }
}

// Verify security answer
async function verifySecurityAnswer() {
  const answerInput = document.getElementById('securityAnswer');
  const answer = answerInput.value.trim();
  const button = document.querySelector('#step2 button[type="submit"]');
  const originalText = button.innerHTML;
  
  // Validate answer
  if (!answer) {
    showError('Please enter your answer', 'securityAnswerHelp');
    answerInput.focus();
    return;
  }
  
  try {
    // Show loading state
    button.disabled = true;
    button.innerHTML = '<span class="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span> Verifying...';
    
    // Call API to verify answer
    const response = await fetch(`${CONFIG.API_BASE_URL}${CONFIG.ENDPOINTS.VERIFY_ANSWER}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        email: state.userEmail,
        answer: answer
      })
    });
    
    const data = await response.json();
    
    if (!response.ok) {
      throw new Error(data.message || 'Incorrect answer. Please try again.');
    }
    
    // Store reset token if provided
    if (data.reset_token) {
      localStorage.setItem('resetToken', data.reset_token);
    }
    
    // Proceed to password reset
    showStep(3);
    
  } catch (error) {
    console.error('Error:', error);
    showError(error.message || 'Failed to verify answer', 'securityAnswerHelp');
    
    // Shake animation for wrong answer
    answerInput.classList.add('is-invalid');
    setTimeout(() => {
      answerInput.classList.remove('is-invalid');
    }, 1000);
    
  } finally {
    // Reset button state
    button.disabled = false;
    button.innerHTML = originalText;
  }
}

// Reset password
async function resetPassword() {
  const newPassword = document.getElementById('newPassword').value;
  const confirmPassword = document.getElementById('confirmPassword').value;
  const button = document.querySelector('#step3 button[type="submit"]');
  const originalText = button.innerHTML;
  
  // Validate passwords
  if (!newPassword || !confirmPassword) {
    showError('Please fill in all fields', 'passwordHelp');
    return;
  }
  
  if (newPassword !== confirmPassword) {
    showError('Passwords do not match', 'passwordHelp');
    return;
  }
  
  // Password strength check (at least 8 characters, 1 number, 1 special char)
  const passwordRegex = /^(?=.*[0-9])(?=.*[!@#$%^&*])[a-zA-Z0-9!@#$%^&*]{8,}$/;
  if (!passwordRegex.test(newPassword)) {
    showError('Password must be at least 8 characters long and contain at least one number and one special character', 'passwordHelp');
    return;
  }
  
  try {
    // Show loading state
    button.disabled = true;
    button.innerHTML = '<span class="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span> Resetting...';
    
    // Get reset token from localStorage or use empty string
    const resetToken = localStorage.getItem('resetToken') || '';
    
    // Call API to reset password
    const response = await fetch(`${CONFIG.API_BASE_URL}${CONFIG.ENDPOINTS.RESET_PASSWORD}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        email: state.userEmail,
        new_password: newPassword,
        reset_token: resetToken
      })
    });
    
    const data = await response.json();
    
    if (!response.ok) {
      throw new Error(data.message || 'Failed to reset password');
    }
    
    // Show success message and redirect to login
    await Swal.fire({
      icon: 'success',
      title: 'Password Reset!',
      text: 'Your password has been reset successfully. Redirecting to login...',
      timer: 3000,
      timerProgressBar: true,
      showConfirmButton: false
    });
    
    // Clear reset token
    localStorage.removeItem('resetToken');
    
    // Redirect to login page
    window.location.href = 'login.html';
    
  } catch (error) {
    console.error('Error:', error);
    showError(error.message || 'Failed to reset password', 'passwordHelp');
    
  } finally {
    // Reset button state
    button.disabled = false;
    button.innerHTML = originalText;
  }
}

// Validate email format
function isValidEmail(email) {
  const re = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  return re.test(email);
}

// Update password strength meter
function updatePasswordStrength(strength) {
  const strengthMeter = document.getElementById('password-strength');
  const strengthText = document.getElementById('password-strength-text');
  
  if (!strengthMeter || !strengthText) return;
  
  let width = 0;
  let color = '#dc3545';
  let text = 'Very Weak';
  
  switch (strength) {
    case 1:
      width = 20;
      color = '#dc3545';
      text = 'Very Weak';
      break;
    case 2:
      width = 40;
      color = '#fd7e14';
      text = 'Weak';
      break;
    case 3:
      width = 60;
      color = '#ffc107';
      text = 'Moderate';
      break;
    case 4:
      width = 80;
      color = '#28a745';
      text = 'Strong';
      break;
    case 5:
      width = 100;
      color = '#20c997';
      text = 'Very Strong';
      break;
    default:
      width = 0;
      text = '';
  }
  
  strengthMeter.style.width = `${width}%`;
  strengthMeter.style.backgroundColor = color;
  strengthText.textContent = text;
  strengthText.style.color = color;
}

// Initialize the form when DOM is fully loaded
document.addEventListener('DOMContentLoaded', function() {
  console.log('Initializing forgot password form');
  
  // Initialize form elements
  const forms = {
    email: document.getElementById('emailForm'),
    security: document.getElementById('securityForm'),
    password: document.getElementById('passwordForm'),
    newPassword: document.getElementById('newPassword'),
    confirmPassword: document.getElementById('confirmPassword')
  };
  
  // Set up form submission handlers
  if (forms.email) {
    forms.email.addEventListener('submit', function(e) {
      e.preventDefault();
      checkEmail();
    });
  }
  
  if (forms.security) {
    forms.security.addEventListener('submit', function(e) {
      e.preventDefault();
      verifySecurityAnswer();
    });
  }
  
  if (forms.password) {
    forms.password.addEventListener('submit', function(e) {
      e.preventDefault();
      resetPassword();
    });
  }
  
  // Set up password strength and confirmation
  if (forms.newPassword) {
    forms.newPassword.addEventListener('input', function() {
      const password = this.value;
      let strength = 0;
      
      if (password.length >= 8) strength++;
      if (/[a-z]/.test(password)) strength++;
      if (/[A-Z]/.test(password)) strength++;
      if (/[0-9]/.test(password)) strength++;
      if (/[^A-Za-z0-9]/.test(password)) strength++;
      
      updatePasswordStrength(strength);
      
      // Update confirmation validation if needed
      if (forms.confirmPassword && forms.confirmPassword.value) {
        validatePasswords();
      }
    });
  }
  
  // Set up password confirmation validation
  if (forms.confirmPassword) {
    forms.confirmPassword.addEventListener('input', function() {
      validatePasswords();
    });
  }
  
  // Show first step
  showStep(1);
  
  console.log('Forgot password form initialized');
});

// Expose necessary functions to global scope
window.checkEmail = checkEmail;
window.verifySecurityAnswer = verifySecurityAnswer;
window.resetPassword = resetPassword;
window.showStep = showStep;
window.togglePassword = togglePassword;
window.backToStep = backToStep;
window.backToStep1 = () => backToStep(1);
window.backToStep2 = () => backToStep(2);};
