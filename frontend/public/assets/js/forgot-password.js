// Global variables
const API_BASE_URL = "http://127.0.0.1:5000";
let currentStep = 1;
let userEmail = "";
let securityQuestion = "";
let emailVerifyBtn = null;
let progressBar = null;
let newPasswordInput = null;
let confirmPasswordInput = null;
let passwordStrength = null;
let passwordStrengthText = null;

// Initialize SweetAlert2 with theme
const Toast = Swal.mixin({
  toast: true,
  position: "top-end",
  showConfirmButton: false,
  timer: 3000,
  timerProgressBar: true,
  didOpen: (toast) => {
    toast.addEventListener("mouseenter", Swal.stopTimer);
    toast.addEventListener("mouseleave", Swal.resumeTimer);
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

// Toggle password visibility with better feedback
function togglePasswordVisibility(inputId, toggleBtn) {
  const input = document.getElementById(inputId);
  if (!input) return;
  
  const icon = toggleBtn?.querySelector('i');
  
  if (input.type === 'password') {
    input.type = 'text';
    if (icon) {
      icon.classList.remove('bi-eye');
      icon.classList.add('bi-eye-slash');
    }
    toggleBtn?.setAttribute('aria-label', 'Hide password');
    toggleBtn?.setAttribute('title', 'Hide password');
  } else {
    input.type = 'password';
    if (icon) {
      icon.classList.remove('bi-eye-slash');
      icon.classList.add('bi-eye');
    }
    toggleBtn?.setAttribute('aria-label', 'Show password');
    toggleBtn?.setAttribute('title', 'Show password');
  }
  
  // Focus back on the input
  input.focus();
}

// Check email and proceed to next step
async function checkEmail(button) {
  console.log('checkEmail function called with button:', button);
  
  // Get email input
  const email = document.getElementById('email');
  if (!email) {
    console.error('Email input element not found');
    Toast.fire({
      icon: 'error',
      title: 'Email input element not found'
    });
    return;
  }

  const emailValue = email.value.trim();
  if (!emailValue) {
    console.error('Email field is empty');
    Toast.fire({
      icon: 'error',
      title: 'Please enter your email address'
    });
    return;
  }

  // Store button reference
  emailVerifyBtn = button;

  // Show loading state
  if (button) {
    button.disabled = true;
    button.innerHTML = '<span class="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span> Verifying...';
  }

  try {
    console.log('Sending request to verify email:', emailValue);
    const response = await fetch(`${API_BASE_URL}/auth/forgot-password`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ email: emailValue })
    });

    if (!response.ok) {
      const error = await response.json();
      console.error('API error:', error);
      throw new Error(error.message || 'Failed to verify email');
    }

    const data = await response.json();
    console.log('API response:', data);
    userEmail = emailValue;
    securityQuestion = data.security_question;

    // Show success message
    Toast.fire({
      icon: 'success',
      title: 'Email verified successfully'
    });

    // Proceed to next step
    showStep(2);
  } catch (error) {
    console.error('Error in checkEmail:', error);
    Toast.fire({
      icon: 'error',
      title: error.message || 'Failed to verify email'
    });
  } finally {
    // Reset button state
    if (button) {
      button.disabled = false;
      button.innerHTML = 'Continue <i class="bi bi-arrow-right ms-2"></i>';
    }
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
  
  resetBtn.disabled = !password || !confirm;
  return false;
}

// Update progress bar based on current step
function updateProgressBar() {
  if (!progressBar) return;
  const progress = ((currentStep - 1) / 2) * 100;
  progressBar.style.width = `${progress}%`;
}

// Show a specific step in the form with smooth transition
function showStep(step) {
  const steps = document.querySelectorAll('.step');
  const currentElement = document.getElementById(`step${currentStep}`);
  const nextElement = document.getElementById(`step${step}`);
  
  // Update step indicators
  document.querySelectorAll('.step-indicator .step').forEach((indicator, index) => {
    if (index + 1 < step) {
      indicator.classList.add('completed');
      indicator.classList.remove('active');
    } else if (index + 1 === step) {
      indicator.classList.add('active');
      indicator.classList.remove('completed');
    } else {
      indicator.classList.remove('active', 'completed');
    }
  });
  
  // Fade out current step
  if (currentElement) {
    currentElement.style.opacity = '0';
    currentElement.style.transition = 'opacity 0.3s ease-in-out';
    
    setTimeout(() => {
      currentElement.style.display = 'none';
      
      // Fade in next step
      if (nextElement) {
        nextElement.style.display = 'block';
        nextElement.style.opacity = '0';
        
        // Trigger reflow
        void nextElement.offsetHeight;
        
        nextElement.style.opacity = '1';
        nextElement.style.transition = 'opacity 0.3s ease-in-out';
      }
    }, 300);
  }
  
  currentStep = step;
  updateProgressBar();
  
  // Focus on first input of the step
  setTimeout(() => {
    const firstInput = nextElement?.querySelector('input');
    if (firstInput) firstInput.focus();
  }, 350);
  
  // Scroll to top of form
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

// Show error message
function showError(message, errorDivId = "error-message") {
  const errorDiv = document.getElementById(errorDivId);
  if (errorDiv) {
    errorDiv.textContent = message;
    errorDiv.classList.remove("d-none");
    // Auto-hide after 5 seconds
    setTimeout(() => {
      errorDiv.classList.add("d-none");
    }, 5000);
  } else {
    console.error("Error div not found:", errorDivId);
    alert(message); // Fallback
  }
}

// Validate email format
function isValidEmail(email) {
  const re = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  return re.test(email);
}

// Step 1: Check email and get security question
async function checkEmail() {
  const emailInput = document.getElementById("email");
  const email = emailInput.value.trim();
  
  // Validate email
  if (!email) {
    showError("Please enter your email address.", "emailHelp");
    emailInput.focus();
    return;
  }

  if (!isValidEmail(email)) {
    showError("Please enter a valid email address.", "emailHelp");
    emailInput.focus();
    return;
  }

  const continueBtn = document.querySelector('#step1 button[onclick="checkEmail()"]');
  const originalBtnText = continueBtn.innerHTML;
  
  try {
    // Show loading state
    continueBtn.disabled = true;
    continueBtn.innerHTML = '<span class="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span> Checking...';
    
    // Call API to check email and get security question
    const response = await fetch(`${API_BASE_URL}/api/auth/forgot-password`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ email }),
    });

    const data = await response.json();
    
    if (!response.ok) {
      throw new Error(data.message || (response.status === 404 
        ? "No account found with this email address." 
        : "An error occurred. Please try again later."));
    }
    
    if (data.success) {
      userEmail = email;
      securityQuestion = data.security_question || "What is your mother's maiden name?";
      document.getElementById("securityQuestion").textContent = securityQuestion;
      showStep(2);
      
      // Show success message
      Toast.fire({
        icon: 'success',
        title: 'Verification email sent!',
        text: 'Please check your email for the verification code.'
      });
    } else {
      throw new Error(data.message || "An error occurred. Please try again.");
    }
  } catch (error) {
    console.error("Error:", error);
    showError(error.message || "An error occurred. Please try again.", "emailHelp");
  } finally {
    // Reset button state
    continueBtn.disabled = false;
    continueBtn.innerHTML = originalBtnText;
  }
}

// Step 2: Verify security answer
async function verifyAnswer() {
  const answerInput = document.getElementById("securityAnswer");
  const answer = answerInput.value.trim();
  
  // Validate answer
  if (!answer) {
    showError("Please enter your answer.", "securityAnswerHelp");
    answerInput.focus();
    return;
  }

  const verifyBtn = document.querySelector('#step2 button[onclick="verifyAnswer()"]');
  const originalBtnText = verifyBtn.innerHTML;
  
  try {
    // Show loading state
    verifyBtn.disabled = true;
    verifyBtn.innerHTML = '<span class="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span> Verifying...';
    
    // Call API to verify security answer
    const response = await fetch(`${API_BASE_URL}/api/auth/verify-answer`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        email: userEmail,
        answer: answer,
      }),
    });

    const data = await response.json();
    
    if (!response.ok) {
      throw new Error(data.message || (response.status === 400 
        ? "Incorrect answer. Please try again." 
        : "An error occurred. Please try again later."));
    }
    
    if (data.success) {
      // Store the reset token if provided
      if (data.reset_token) {
        localStorage.setItem('resetToken', data.reset_token);
      }
      
      showStep(3);
      
      // Show success message
      Toast.fire({
        icon: 'success',
        title: 'Identity verified!',
        text: 'You can now set a new password for your account.'
      });
    } else {
      throw new Error(data.message || "An error occurred. Please try again.");
    }
  } catch (error) {
    console.error("Error:", error);
    showError(error.message || "An error occurred. Please try again.", "securityAnswerHelp");
    
    // Shake animation for wrong answer
    answerInput.classList.add('is-invalid');
    setTimeout(() => {
      answerInput.classList.remove('is-invalid');
    }, 1000);
  } finally {
    // Reset button state
    verifyBtn.disabled = false;
    verifyBtn.innerHTML = originalBtnText;
  }
}

// Navigation functions
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

// Expose functions to global scope
window.checkEmail = checkEmail;
window.verifyAnswer = verifyAnswer;
window.resetPassword = resetPassword;
window.backToStep = backToStep;
window.backToStep1 = () => backToStep(1);
window.backToStep2 = () => backToStep(2);

// Initialize the form
document.addEventListener("DOMContentLoaded", () => {
  console.log('Initializing forgot password form');
  
  // Query DOM elements
  progressBar = document.getElementById('progress-bar');
  console.log('Found progress bar:', progressBar !== null);
  
  newPasswordInput = document.getElementById('newPassword');
  console.log('Found new password input:', newPasswordInput !== null);
  
  confirmPasswordInput = document.getElementById('confirmPassword');
  console.log('Found confirm password input:', confirmPasswordInput !== null);
  
  passwordStrength = document.getElementById('password-strength');
  console.log('Found password strength:', passwordStrength !== null);
  
  passwordStrengthText = document.querySelector('#password-strength-text span');
  console.log('Found password strength text:', passwordStrengthText !== null);
  
  // Initialize tooltips
  const tooltipTriggers = document.querySelectorAll('[data-bs-toggle="tooltip"]');
  console.log('Found tooltip triggers:', tooltipTriggers.length);
  tooltipTriggers.forEach(tooltipTriggerEl => {
    new bootstrap.Tooltip(tooltipTriggerEl);
  });

  // Initialize form state
  showStep(1);
  updateProgressBar();

  // Expose functions to global scope
  window.checkEmail = checkEmail;
  console.log('Exposing checkEmail function:', typeof window.checkEmail === 'function');
  
  window.verifyAnswer = verifyAnswer;
  console.log('Exposing verifyAnswer function:', typeof window.verifyAnswer === 'function');
  
  window.resetPassword = resetPassword;
  console.log('Exposing resetPassword function:', typeof window.resetPassword === 'function');
  
  window.backToStep = backToStep;
  console.log('Exposing backToStep function:', typeof window.backToStep === 'function');
  
  window.backToStep1 = () => backToStep(1);
  console.log('Exposing backToStep1 function:', typeof window.backToStep1 === 'function');
  
  window.backToStep2 = () => backToStep(2);
  console.log('Exposing backToStep2 function:', typeof window.backToStep2 === 'function');

  console.log('Initialization complete');
});
