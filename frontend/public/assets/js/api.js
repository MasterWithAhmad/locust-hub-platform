// API base URL configuration
let API_BASE_URL;

// Determine the environment and set the appropriate API URL
const isLocalhost = window.location.hostname === 'localhost' || 
                   window.location.hostname === '127.0.0.1' ||
                   window.location.hostname === '';

if (isLocalhost) {
    // Development environment - use the same origin as the frontend
    API_BASE_URL = window.location.origin.replace(/\/+$/, '');
    
    // If the frontend is running on port 3000, assume backend is on 5000
    if (window.location.port === '3000') {
        API_BASE_URL = API_BASE_URL.replace(':3000', ':5000');
    }
} else {
    // Production environment - use the same origin
    API_BASE_URL = window.location.origin.replace(/\/+$/, '');
}

// Ensure we don't have double slashes in the URL
API_BASE_URL = API_BASE_URL.replace(/([^:])\/+/g, '$1/');

// Expose to window for global access
window.API_BASE_URL = API_BASE_URL;

console.log('Frontend Origin:', window.location.origin);
console.log('API Base URL:', API_BASE_URL);

// Helper function to handle API responses
async function handleResponse(response) {
    // First, clone the response so we can read it multiple times
    const responseClone = response.clone();
    let errorData = {};
    
    try {
        errorData = await responseClone.json().catch(() => ({}));
    } catch (e) {
        console.error('Error parsing error response:', e);
    }
    
    // Handle 401 Unauthorized
    if (response.status === 401) {
        const url = response.url || '';
        // For password change endpoint, throw a specific error
        if (url.endsWith('/user/password')) {
            throw new Error(errorData.error || 'Invalid current password. Please try again.');
        }
        // For other endpoints, log the user out
        localStorage.removeItem('token');
        localStorage.removeItem('user');
        window.location.href = '/login.html';
        return Promise.reject(new Error('Session expired. Please log in again.'));
    }

    // Handle other error statuses
    if (!response.ok) {
        const error = new Error(errorData.error || `HTTP error! status: ${response.status}`);
        error.status = response.status;
        error.response = errorData;
        throw error;
    }
    return response.json();
}

// Authentication functions
async function register(fullName, email, password, securityQuestion, securityAnswer) {
    const response = await fetch(`${API_BASE_URL}/api/register`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            full_name: fullName,
            email,
            password,
            security_question: securityQuestion,
            security_answer: securityAnswer
        }),
    });
    return handleResponse(response);
}

async function login(email, password) {
    try {
        console.log('Attempting login with email:', email);
        const response = await fetch(`${API_BASE_URL}/api/login`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            },
            body: JSON.stringify({ email, password }),
            credentials: 'include'  // Important for cookies/session
        });
        
        console.log('Login response status:', response.status);

        const data = await handleResponse(response);

        // Debug logging
        console.log('Login response data:', data);

        if (!data.access_token) {
            throw new Error('No access token received from server');
        }

        // Verify user data exists in the response
        if (!data.user || !data.user.id) {
            console.error('Invalid user data in login response:', data);
            throw new Error('Invalid user data received from server');
        }

        // Ensure the user ID is a string
        const userData = {
            ...data.user,
            id: String(data.user.id)  // Ensure ID is a string
        };

        // Store the token and user data
        localStorage.setItem('token', data.access_token);
        localStorage.setItem('user', JSON.stringify(userData));

        console.log('Login successful, user data stored:', userData);
        return {
            ...data,
            user: userData
        };
    } catch (error) {
        console.error('Login error:', error);
        throw error; // Re-throw to allow calling function to handle it
    }
}

function logout() {
    localStorage.removeItem('token');
    localStorage.removeItem('user');
    window.location.href = '/login.html';
}

function isLoggedIn() {
    return !!localStorage.getItem('token');
}

function getCurrentUser() {
    try {
        const userJson = localStorage.getItem('user');
        if (!userJson) {
            console.log('No user data found in localStorage');
            return null;
        }
        const user = JSON.parse(userJson);
        if (!user || typeof user !== 'object') {
            console.error('Invalid user data in localStorage:', user);
            return null;
        }
        console.log('Current user from localStorage:', user);
        return user;
    } catch (error) {
        console.error('Error parsing user data from localStorage:', error);
        return null;
    }
}

function getAuthHeader() {
    try {
        const token = localStorage.getItem('token');
        const user = getCurrentUser();

        if (!token) {
            console.warn('No authentication token found');
            throw new Error('No authentication token found');
        }

        // Log token info for debugging (don't log the actual token for security)
        console.log('Auth token exists:', token ? 'Yes' : 'No');
        console.log('Current user ID:', user?.id || 'No user ID');

        // Always return the token in the Authorization header
        return {
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            'Authorization': `Bearer ${token}`
        };

    } catch (error) {
        console.error('Error in getAuthHeader:', error);
        // Return headers without Authorization on error
        return {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        };
    }
}

// Contact form function
async function sendContactMessage({ name, email, phone, message }) {
    const response = await fetch(`${API_BASE_URL}/contact`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        },
        body: JSON.stringify({ name, email, phone, message })
    });
    return handleResponse(response);
}

// Prediction functions
async function savePrediction(predictionData) {
    const user = getCurrentUser();
    const response = await fetch(`${API_BASE_URL}/api/save_prediction`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'Authorization': JSON.stringify({
                isLoggedIn: true,
                email: user.email
            })
        },
        body: JSON.stringify(predictionData)
    });
    return handleResponse(response);
}

async function getAllPredictions() {
    try {
        console.log('Fetching predictions from:', `${API_BASE_URL}/predictions`);

        // Get the current user to verify authentication
        const user = getCurrentUser();
        if (!user || !user.id) {
            throw new Error('User not authenticated');
        }

        const headers = getAuthHeader();
        console.log('Request headers:', headers);

        const response = await fetch(`${API_BASE_URL}/api/predictions`, {
            method: 'GET',
            headers: {
                ...headers,
                'Accept': 'application/json',
                'Cache-Control': 'no-cache',
                'Pragma': 'no-cache'
            },
            credentials: 'same-origin'
        });

        console.log('Response status:', response.status);

        if (!response.ok) {
            const errorText = await response.text();
            console.error('Error response:', errorText);
            let errorData;
            try {
                errorData = errorText ? JSON.parse(errorText) : { message: 'Unknown error' };
            } catch (e) {
                errorData = { error: errorText || 'Unknown error occurred' };
            }
            console.error('Error response:', errorData);
            const error = new Error(errorData.error || `HTTP error! status: ${response.status}`);
            error.status = response.status;
            throw error;
        }

        const data = await response.json();
        console.log('Predictions data:', data);
        return data;
    } catch (error) {
        console.error('Error in getAllPredictions:', error);
        if (!error.status) error.status = 0; // Network error
        throw error;
    }
}

async function deletePrediction(predictionId) {
    const response = await fetch(`${API_BASE_URL}/api/predictions/${predictionId}`, {
        method: 'DELETE',
        headers: getAuthHeader()
    });
    return handleResponse(response);
}

async function predict(predictionInputData) {
    const body = {
        REGION: predictionInputData.REGION,
        COUNTRYNAME: predictionInputData.COUNTRYNAME,
        STARTYEAR: predictionInputData.STARTYEAR,
        STARTMONTH: predictionInputData.STARTMONTH,
        PPT: predictionInputData.PPT,
        TMAX: predictionInputData.TMAX,
        SOILMOISTURE: predictionInputData.SOILMOISTURE
    };
    if (predictionInputData.MODEL_NAME) {
        body.MODEL_NAME = predictionInputData.MODEL_NAME;
    }
    const response = await fetch(`${API_BASE_URL}/api/predict`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            ...getAuthHeader(),
        },
        body: JSON.stringify(body),
    });
    return handleResponse(response);
}

// Options functions
async function getOptions() {
    console.log('Calling getOptions API...');
    const response = await fetch(`${API_BASE_URL}/api/options`, {
        method: 'GET',
        headers: {
            'Content-Type': 'application/json',
            ...getAuthHeader(),
        },
    });
    console.log('Options API response:', response);
    return handleResponse(response);
}

// Analytics functions
async function getPredictionSummary() {
    console.log('Calling getPredictionSummary API...');
    const response = await fetch(`${API_BASE_URL}/api/analytics/prediction_summary`, {
        method: 'GET',
        headers: {
            'Content-Type': 'application/json',
            ...getAuthHeader(),
        },
    });
    console.log('Prediction Summary API response:', response);
    return handleResponse(response);
}

async function getPredictionsOverTime() {
    console.log('Calling getPredictionsOverTime API...');
    const response = await fetch(`${API_BASE_URL}/api/analytics/predictions_over_time`, {
        method: 'GET',
        headers: {
            'Content-Type': 'application/json',
            ...getAuthHeader(),
        },
    });
    console.log('Predictions Over Time API response:', response);
    return handleResponse(response);
}

async function getPredictionsByLocation() {
    console.log('Calling getPredictionsByLocation API...');
    const response = await fetch(`${API_BASE_URL}/api/analytics/predictions_by_location`, {
        method: 'GET',
        headers: {
            'Content-Type': 'application/json',
            ...getAuthHeader(),
        },
    });
    console.log('Predictions By Location API response:', response);
    return handleResponse(response);
}

async function getEnvironmentalFactorsSummary() {
    console.log('Calling getEnvironmentalFactorsSummary API...');
    const response = await fetch(`${API_BASE_URL}/api/analytics/environmental_factors_summary`, {
        method: 'GET',
        headers: {
            'Content-Type': 'application/json',
            ...getAuthHeader(),
        },
    });
    console.log('Environmental Factors Summary API response:', response);
    return handleResponse(response);
}

async function getFeedbackAnalytics() {
    console.log('Calling getFeedbackAnalytics API...');
    const response = await fetch(`${API_BASE_URL}/api/analytics/feedback`, {
        method: 'GET',
        headers: {
            'Content-Type': 'application/json',
            ...getAuthHeader(),
        },
    });
    console.log('Feedback Analytics API response:', response);
    return handleResponse(response);
}

async function submitPredictionFeedback(predictionId, feedback) {
    const response = await fetch(`${API_BASE_URL}/api/predictions/${predictionId}/feedback`, {
        method: 'POST',
        headers: getAuthHeader(),
        body: JSON.stringify({ feedback })
    });
    return handleResponse(response);
}

// Export the API functions
window.api = {
    auth: {
        register,
        login,
        logout,
        isLoggedIn,
        getCurrentUser,
        async requestPasswordReset(email) {
            const response = await fetch('/api/auth/request-password-reset', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ email })
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.message || 'Failed to request password reset');
            }

            return response.json();
        },
        async resetPassword(token, newPassword) {
            const response = await fetch('/api/auth/reset-password', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ token, newPassword })
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.message || 'Failed to reset password');
            }

            return response.json();
        },
        async getSecurityQuestion(email) {
            const response = await fetch(`${API_BASE_URL}/auth/security-question`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ email })
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.message || 'Failed to get security question');
            }

            return response.json();
        },
        async verifySecurityAnswer(email, answer) {
            const response = await fetch(`${API_BASE_URL}/auth/verify-answer`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ email, answer })
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.message || 'Failed to verify security answer');
            }

            return response.json();
        },
        async resetPasswordWithEmail(email, newPassword) {
            const response = await fetch(`${API_BASE_URL}/auth/reset-password-with-email`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ email, new_password: newPassword })
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.message || 'Failed to reset password with email');
            }

            return response.json();
        }
    },
    predictions: {
        save: savePrediction,
        getAll: getAllPredictions,
        delete: deletePrediction,
        submitFeedback: submitPredictionFeedback
    },
    options: {
        getOptions
    },
    predict,
    analytics: {
        getPredictionSummary,
        getPredictionsOverTime,
        getPredictionsByLocation,
        getEnvironmentalFactorsSummary,
        getFeedbackAnalytics,
    },
    user: {
        getUserDetails: async function () {
            console.log('Calling getUserDetails API...');
            const response = await fetch(`${API_BASE_URL}/api/user`, {
                method: 'GET',
                headers: {
                    'Content-Type': 'application/json',
                    ...getAuthHeader(),
                },
            });
            console.log('User Details API response:', response);
            return handleResponse(response);
        },
        updateProfile: async function (profileData) {
            console.log('Calling updateProfile API...', profileData);
            const response = await fetch(`${API_BASE_URL}/user/profile`, {
                method: 'PUT',
                headers: {
                    'Content-Type': 'application/json',
                    ...getAuthHeader(),
                },
                body: JSON.stringify(profileData)
            });
            console.log('Update Profile API response:', response);
            return handleResponse(response);
        },
        changePassword: async function (passwordData) {
            console.log('Calling changePassword API...', passwordData);
            const response = await fetch(`${API_BASE_URL}/user/password`, {
                method: 'PUT',
                headers: {
                    'Content-Type': 'application/json',
                    ...getAuthHeader(),
                },
                body: JSON.stringify(passwordData)
            });
            console.log('Change Password API response:', response);
            return handleResponse(response);
        },
        deleteAccount: async function () {
            console.log('Calling deleteAccount API...');
            try {
                const response = await fetch(`${API_BASE_URL}/account/delete`, {
                    method: 'DELETE',
                    headers: {
                        'Content-Type': 'application/json',
                        ...getAuthHeader(),
                    },
                    credentials: 'include'
                });

                const data = await response.json().catch(() => ({}));

                if (response.status === 401) {
                    // Token expired or invalid, logout and redirect
                    logout();
                    return { success: false, message: 'Session expired. Please log in again.' };
                }

                if (!response.ok) {
                    throw new Error(data.message || 'Failed to delete account');
                }

                // Clear all auth data from localStorage
                localStorage.removeItem('token');
                localStorage.removeItem('user');

                return {
                    success: true,
                    ...data,
                    redirect: '/login.html?accountDeleted=true'  // Add redirect URL
                };

            } catch (error) {
                console.error('Error in deleteAccount:', error);
                throw error;
            }
        }
    },
    blog: {
        // Get all blog posts (authenticated user only)
        getPosts: async function() {
            const response = await fetch(`${API_BASE_URL}/api/users/me/blogposts`, {
                headers: getAuthHeader(),
                credentials: 'include'
            });
            return handleResponse(response);
        },
        
        // Get all public blog posts (no auth required)
        getPublicPosts: async () => {
            const url = `${API_BASE_URL}/api/blogposts/public`;
            console.log('Fetching public posts from:', url);
            
            const response = await fetch(url, {
                method: 'GET',
                headers: {
                    'Content-Type': 'application/json',
                    'Accept': 'application/json'
                },
                credentials: 'include'
            });
            
            console.log('Response status:', response.status);
            return handleResponse(response);
        },
        
        // Get a single blog post by ID
        getPost: async (postId) => {
            try {
                console.log(`Fetching blog post with ID: ${postId}`);
                const url = `${API_BASE_URL}/api/blogposts/${postId}`;
                console.log('API URL:', url);
                
                const headers = getAuthHeader();
                console.log('Request headers:', headers);
                
                const response = await fetch(url, {
                    method: 'GET',
                    headers: headers,
                    credentials: 'include'
                });
                
                console.log('Response status:', response.status);
                const data = await handleResponse(response);
                console.log('Blog post data received:', data);
                return data;
            } catch (error) {
                console.error('Error in getPost API call:', error);
                throw error; // Re-throw to be handled by the caller
            }
        },
        
        // Create a new blog post
        createPost: async (postData) => {
            const response = await fetch(`${API_BASE_URL}/api/blogposts`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    ...getAuthHeader()
                },
                credentials: 'include',
                body: JSON.stringify(postData)
            });
            return handleResponse(response);
        },
        
        // Update an existing blog post
        updatePost: async (postId, postData) => {
            const response = await fetch(`${API_BASE_URL}/api/blogposts/${postId}`, {
                method: 'PUT',
                headers: {
                    'Content-Type': 'application/json',
                    ...getAuthHeader()
                },
                credentials: 'include',
                body: JSON.stringify(postData)
            });
            return handleResponse(response);
        },
        
        // Delete a blog post
        deletePost: async (postId) => {
            const response = await fetch(`${API_BASE_URL}/api/blogposts/${postId}`, {
                method: 'DELETE',
                headers: getAuthHeader(),
                credentials: 'include'
            });
            return handleResponse(response);
        },
        
        // Upload an image for a blog post
        uploadImage: async (file) => {
            if (!file) {
                console.error('No file provided for upload');
                throw new Error('No file provided for upload');
            }
            
            // Validate file size (server also validates, but we'll check client-side first)
            const maxSize = 15 * 1024 * 1024; // 15MB (slightly less than server limit)
            if (file.size > maxSize) {
                const error = new Error(`File is too large. Maximum size is ${maxSize / 1024 / 1024}MB`);
                error.code = 'FILE_TOO_LARGE';
                throw error;
            }
            
            console.log('Preparing to upload file:', file.name, 'Size:', file.size, 'bytes');
            
            const formData = new FormData();
            formData.append('image', file, file.name);
            
            // Get auth headers but remove Content-Type to let the browser set it with the correct boundary
            const headers = getAuthHeader();
            delete headers['Content-Type'];
            
            // Add a timeout to prevent hanging requests
            const controller = new AbortController();
            const timeoutMs = 120000; // 2 minutes for large files
            const timeoutId = setTimeout(() => controller.abort(), timeoutMs);
            
            try {
                console.log(`Sending upload request to server (timeout: ${timeoutMs/1000}s)...`);
                
                // Show upload progress if possible
                const xhr = new XMLHttpRequest();
                
                const uploadPromise = new Promise((resolve, reject) => {
                    xhr.open('POST', `${API_BASE_URL}/api/blogposts/upload`, true);
                    
                    // Set request headers
                    Object.entries(headers).forEach(([key, value]) => {
                        xhr.setRequestHeader(key, value);
                    });
                    
                    xhr.withCredentials = true;
                    
                    xhr.upload.onprogress = (event) => {
                        if (event.lengthComputable) {
                            const percentComplete = Math.round((event.loaded / event.total) * 100);
                            console.log(`Upload progress: ${percentComplete}%`);
                            // You can update a progress bar here if needed
                        }
                    };
                    
                    xhr.onload = function() {
                        clearTimeout(timeoutId);
                        
                        if (xhr.status >= 200 && xhr.status < 300) {
                            try {
                                const response = JSON.parse(xhr.responseText);
                                console.log('Upload successful:', response);
                                resolve(response);
                            } catch (e) {
                                console.error('Error parsing upload response:', e);
                                reject(new Error('Invalid server response'));
                            }
                        } else {
                            let errorMessage = `Upload failed with status ${xhr.status}`;
                            try {
                                const errorData = JSON.parse(xhr.responseText);
                                errorMessage = errorData.error || errorData.message || errorMessage;
                            } catch (e) {
                                // Couldn't parse error response
                            }
                            console.error('Upload failed:', errorMessage);
                            const error = new Error(errorMessage);
                            error.status = xhr.status;
                            reject(error);
                        }
                    };
                    
                    xhr.onerror = function() {
                        clearTimeout(timeoutId);
                        console.error('Network error during upload');
                        reject(new Error('Network error. Please check your connection and try again.'));
                    };
                    
                    xhr.ontimeout = function() {
                        console.error('Upload timed out');
                        controller.abort();
                        reject(new Error('Upload timed out. The server took too long to respond.'));
                    };
                    
                    // Send the form data
                    xhr.send(formData);
                });
                
                // Return the upload promise
                return await uploadPromise;
                
            } catch (error) {
                clearTimeout(timeoutId);
                console.error('Upload error:', error);
                
                // Handle specific error cases
                if (error.name === 'AbortError' || error.code === 'ABORT_ERR') {
                    throw new Error('Upload was cancelled or timed out. Please try again.');
                } else if (error.code === 'FILE_TOO_LARGE') {
                    throw error; // Already has a good message
                } else if (error.message.includes('NetworkError')) {
                    throw new Error('Network error. Please check your connection and try again.');
                } else if (error.status === 413) {
                    throw new Error('File is too large. Please choose a smaller file.');
                } else if (error.status === 415) {
                    throw new Error('File type not supported. Please upload an image file (JPEG, PNG, GIF, or WebP).');
                } else if (error.status === 507) {
                    throw new Error('Server is out of storage space. Please try again later or contact support.');
                }
                
                // For other errors, use the server's error message or a generic one
                throw new Error(error.message || 'Failed to upload file. Please try again.');
            }
        }
    }
};