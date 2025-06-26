// Global variables
const API_BASE_URL = "http://127.0.0.1:5000";
let currentStep = 1;
let userEmail = "";

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
  const toggleNewPassword = document.querySelector("#toggleNewPassword");
  const newPassword = document.querySelector("#newPassword");
  const toggleConfirmPassword = document.querySelector(
    "#toggleConfirmPassword"
  );
  const confirmPassword = document.querySelector("#confirmPassword");

  // Toggle new password visibility
  if (toggleNewPassword && newPassword) {
    toggleNewPassword.addEventListener("click", function () {
      togglePasswordVisibility(newPassword, this);
    });
  }

  // Toggle confirm password visibility
  if (toggleConfirmPassword && confirmPassword) {
    toggleConfirmPassword.addEventListener("click", function () {
      togglePasswordVisibility(confirmPassword, this);
    });
  }

  // Show initial step
  showStep(1);
});

// Show a specific step in the form
function showStep(step) {
  // Hide all steps
  document.querySelectorAll(".step").forEach((div) => {
    div.style.display = "none";
  });

  // Show the current step
  const currentStepElement = document.getElementById(`step${step}`);
  if (currentStepElement) {
    currentStepElement.style.display = "block";
  }

  currentStep = step;

  // Scroll to top of form
  window.scrollTo({ top: 0, behavior: "smooth" });
}

// Toggle password visibility
function togglePasswordVisibility(inputId, toggleBtn) {
  const input = document.getElementById(inputId);
  const icon = toggleBtn.querySelector("i");

  if (input && icon) {
    if (input.type === "password") {
      input.type = "text";
      icon.classList.remove("bi-eye");
      icon.classList.add("bi-eye-slash");
    } else {
      input.type = "password";
      icon.classList.remove("bi-eye-slash");
      icon.classList.add("bi-eye");
    }
  }
}

// Initialize the form
document.addEventListener("DOMContentLoaded", () => {
  showStep(1);

  // Add event listeners for password toggles
  const toggleNewPassword = document.getElementById("toggleNewPassword");
  const toggleConfirmPassword = document.getElementById(
    "toggleConfirmPassword"
  );

  if (toggleNewPassword) {
    toggleNewPassword.addEventListener("click", function () {
      togglePasswordVisibility("newPassword", this);
    });
  }

  if (toggleConfirmPassword) {
    toggleConfirmPassword.addEventListener("click", function () {
      togglePasswordVisibility("confirmPassword", this);
    });
  }

  // Real-time password match validation
  const confirmPasswordInput = document.getElementById("confirmPassword");
  if (confirmPasswordInput) {
    confirmPasswordInput.addEventListener("input", function () {
      const newPassword = document.getElementById("newPassword");
      if (newPassword && this.value && newPassword.value !== this.value) {
        this.setCustomValidity("Passwords do not match");
      } else {
        this.setCustomValidity("");
      }
    });
  }
});

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
  const email = document.getElementById("email").value.trim();
  const continueBtn = document.querySelector(
    '#step1 button[onclick="checkEmail()"]'
  );

  if (!email) {
    await Swal.fire({
      icon: "error",
      title: "Error",
      text: "Please enter your email address",
      confirmButtonText: "OK",
      confirmButtonColor: "#0d6efd",
    });
    return;
  }

  if (!isValidEmail(email)) {
    await Swal.fire({
      icon: "error",
      title: "Invalid Email",
      text: "Please enter a valid email address",
      confirmButtonText: "OK",
      confirmButtonColor: "#0d6efd",
    });
    return;
  }

  try {
    // Show loading state
    Swal.fire({
      title: "Verifying Email",
      text: "Please wait while we check your email...",
      allowOutsideClick: false,
      didOpen: () => {
        Swal.showLoading();
      },
    });

    const response = await fetch(`${API_BASE_URL}/api/auth/forgot-password`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Accept: "application/json",
      },
      body: JSON.stringify({ email }),
      credentials: "include", // Important for sending cookies with CORS
    });

    const data = await response.json();

    // Close any open dialogs
    Swal.close();

    if (!response.ok) {
      throw new Error(data.error || "Failed to verify email");
    }

    if (!data.security_question) {
      throw new Error("No security question found for this account");
    }

    // Store email for next steps
    userEmail = email;

    // Show security question immediately
    document.getElementById("securityQuestion").textContent =
      data.security_question;
    showStep(2);
  } catch (error) {
    console.error("Error:", error);
    await Swal.fire({
      icon: "error",
      title: "Error",
      text: error.message || "An error occurred while verifying your email",
      confirmButtonText: "OK",
      confirmButtonColor: "#0d6efd",
    });
  }
}

// Step 2: Verify security answer
async function verifyAnswer() {
  const answerInput = document.getElementById("securityAnswer");
  const answer = answerInput ? answerInput.value.trim() : "";
  const submitBtn = document.querySelector('#step2 button[type="button"]');

  if (!answer) {
    await Swal.fire({
      icon: "error",
      title: "Error",
      text: "Please enter your answer",
      confirmButtonText: "OK",
      confirmButtonColor: "#0d6efd",
    });
    if (answerInput) answerInput.focus();
    return;
  }

  try {
    // Show loading state
    Swal.fire({
      title: "Verifying Answer",
      text: "Please wait while we verify your answer...",
      allowOutsideClick: false,
      didOpen: () => {
        Swal.showLoading();
      },
    });

    // Call API to verify answer
    const response = await fetch(`${API_BASE_URL}/api/auth/verify-answer`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Accept: "application/json",
      },
      body: JSON.stringify({
        email: userEmail,
        answer: answer,
      }),
      credentials: "include",
    });

    const data = await response.json();

    // Close loading dialog
    Swal.close();

    if (!response.ok) {
      throw new Error(data.error || "Verification failed. Please try again.");
    }

    // Show success message
    await Swal.fire({
      icon: "success",
      title: "Success!",
      text: "Your answer has been verified. You can now set a new password.",
      confirmButtonText: "Continue",
      confirmButtonColor: "#0d6efd",
    });

    // Show password reset form
    showStep(3);
  } catch (error) {
    console.error("Error:", error);
    await Swal.fire({
      icon: "error",
      title: "Error",
      text: error.message || "An error occurred while verifying your answer",
      confirmButtonText: "OK",
      confirmButtonColor: "#0d6efd",
    });
  } finally {
    // Reset button state if it exists
    if (submitBtn) {
      submitBtn.disabled = false;
      submitBtn.innerHTML = 'Verify <i class="bi bi-shield-check ms-2"></i>';
    }
  }
}

// Step 3: Reset password
async function resetPassword() {
  const newPassword = document.getElementById("newPassword").value;
  const confirmPassword = document.getElementById("confirmPassword").value;
  const submitBtn = document.querySelector('#step3 button[type="button"]');

  // Log the start of the reset process
  console.log("Starting password reset process...");
  console.log("User email:", userEmail);
  console.log("New password length:", newPassword.length);

  // Validate passwords
  if (newPassword.length < 8) {
    console.log("Password validation failed: too short");
    await Swal.fire({
      icon: "error",
      title: "Password Too Short",
      text: "Password must be at least 8 characters long",
      confirmButtonText: "OK",
      confirmButtonColor: "#0d6efd",
    });
    return;
  }

  if (newPassword !== confirmPassword) {
    console.log("Password validation failed: passwords do not match");
    await Swal.fire({
      icon: "error",
      title: "Passwords Do Not Match",
      text: "Please make sure both passwords match",
      confirmButtonText: "OK",
      confirmButtonColor: "#0d6efd",
    });
    return;
  }

  try {
    // Update button state
    if (submitBtn) {
      submitBtn.disabled = true;
      submitBtn.innerHTML =
        '<span class="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span>Resetting...';
    }

    // Show loading state
    Swal.fire({
      title: "Resetting Password",
      text: "Please wait while we update your password...",
      allowOutsideClick: false,
      didOpen: () => {
        Swal.showLoading();
      },
    });

    // Prepare request data
    const requestData = {
      email: userEmail,
      new_password: newPassword,
    };

    console.log("Sending reset password request:", {
      url: `${API_BASE_URL}/api/auth/reset-password`,
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Accept: "application/json",
      },
      body: requestData,
      credentials: "include",
    });

    // Call API to reset password
    const response = await fetch(`${API_BASE_URL}/api/auth/reset-password`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Accept: "application/json",
      },
      body: JSON.stringify(requestData),
      credentials: "include",
    });

    console.log("Received response status:", response.status);

    let data;
    try {
      const responseText = await response.text();
      console.log("Raw response text:", responseText);
      try {
        data = JSON.parse(responseText);
        console.log("Response data:", data);
      } catch (e) {
        console.error("Failed to parse JSON string:", e);
        throw new Error("Invalid JSON response from server");
      }
    } catch (e) {
      console.error("Error getting response text:", e);
      throw new Error("Could not read response from server");
    }

    // Close any open dialogs
    Swal.close();

    if (!response.ok) {
      console.error("Error response from server:", data);
      throw new Error(
        data.error || `Failed to reset password (${response.status})`
      );
    }

    console.log("Password reset successful, showing success message");

    // Show success message
    await Swal.fire({
      icon: "success",
      title: "Password Reset Successful",
      text:
        data.message ||
        "Your password has been reset successfully. You can now log in with your new password.",
      confirmButtonText: "Go to Login",
      confirmButtonColor: "#0d6efd",
      allowOutsideClick: false,
    });

    // Redirect to login page
    console.log("Redirecting to login page...");
    window.location.href = "login.html";
  } catch (error) {
    console.error("Error in resetPassword:", error);

    // Close any open dialogs
    Swal.close();

    // Show error message
    await Swal.fire({
      icon: "error",
      title: "Error",
      text: error.message || "An error occurred while resetting your password",
      confirmButtonText: "OK",
      confirmButtonColor: "#0d6efd",
    });

    // Reset button state
    if (submitBtn) {
      submitBtn.disabled = false;
      submitBtn.innerHTML =
        'Reset Password <i class="bi bi-arrow-repeat ms-2"></i>';
    }
  }
}
// Navigation functions
function backToStep(step) {
  showStep(step);
}

function backToStep1() {
  showStep(1);
}
function backToStep2() {
  showStep(2);
}

// Initialize the form
document.addEventListener("DOMContentLoaded", () => {
  showStep(1);

  // Allow form submission with Enter key
  document.getElementById("email").addEventListener("keypress", (e) => {
    if (e.key === "Enter") requestReset();
  });

  document
    .getElementById("securityAnswer")
    .addEventListener("keypress", (e) => {
      if (e.key === "Enter") verifyAnswer();
    });

  document.getElementById("newPassword").addEventListener("keypress", (e) => {
    if (e.key === "Enter") resetPassword();
  });

  document
    .getElementById("confirmPassword")
    .addEventListener("keypress", (e) => {
      if (e.key === "Enter") resetPassword();
    });
});
