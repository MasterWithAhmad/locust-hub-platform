// API base URL - use the current origin to avoid CORS issues
const API_BASE_URL = window.location.origin.includes('3000') 
  ? 'http://localhost:5000/api' 
  : `${window.location.origin}/api`;

console.log('API Base URL:', API_BASE_URL);

// Helper function to handle API responses
async function handleResponse(response) {
    if (response.status === 401) {
        // Token expired or invalid, redirect to login
        localStorage.removeItem('token');
        localStorage.removeItem('user');
        window.location.href = '/login.html';
    }
    
    if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
    }
    return response.json();
}

// Authentication functions
async function register(fullName, email, password, securityQuestion, securityAnswer) {
    const response = await fetch(`${API_BASE_URL}/register`, {
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
        const response = await fetch(`${API_BASE_URL}/login`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            },
            body: JSON.stringify({ email, password }),
        });

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

// Prediction functions
async function savePrediction(predictionData) {
    const user = getCurrentUser();
    const response = await fetch(`${API_BASE_URL}/save_prediction`, {
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
        
        const response = await fetch(`${API_BASE_URL}/predictions`, {
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
    const response = await fetch(`${API_BASE_URL}/predictions/${predictionId}`, {
        method: 'DELETE',
        headers: getAuthHeader()
    });
    return handleResponse(response);
}

async function predict(predictionInputData) {
    const response = await fetch(`${API_BASE_URL}/predict`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            ...getAuthHeader(),
        },
        body: JSON.stringify({
            REGION: predictionInputData.REGION,
            COUNTRYNAME: predictionInputData.COUNTRYNAME,
            STARTYEAR: predictionInputData.STARTYEAR,
            STARTMONTH: predictionInputData.STARTMONTH,
            PPT: predictionInputData.PPT,
            TMAX: predictionInputData.TMAX,
            SOILMOISTURE: predictionInputData.SOILMOISTURE
        }),
    });
    return handleResponse(response);
}

// Options functions
async function getOptions() {
    console.log('Calling getOptions API...');
    const response = await fetch(`${API_BASE_URL}/options`, {
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
    const response = await fetch(`${API_BASE_URL}/analytics/prediction_summary`, {
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
    const response = await fetch(`${API_BASE_URL}/analytics/predictions_over_time`, {
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
    const response = await fetch(`${API_BASE_URL}/analytics/predictions_by_location`, {
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
    const response = await fetch(`${API_BASE_URL}/analytics/environmental_factors_summary`, {
        method: 'GET',
        headers: {
            'Content-Type': 'application/json',
            ...getAuthHeader(),
        },
    });
    console.log('Environmental Factors Summary API response:', response);
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
        delete: deletePrediction
    },
    options: {
        getOptions
    },
    predict,
    analytics: {
        getPredictionSummary,
        getPredictionsOverTime,
        getPredictionsByLocation,
        getEnvironmentalFactorsSummary
    },
    user: {
        getUserDetails: async function() {
            console.log('Calling getUserDetails API...');
            const response = await fetch(`${API_BASE_URL}/user`, {
                method: 'GET',
                headers: {
                    'Content-Type': 'application/json',
                    ...getAuthHeader(),
                },
            });
            console.log('User Details API response:', response);
            return handleResponse(response);
        },
        updateProfile: async function(profileData) {
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
        changePassword: async function(passwordData) {
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
        deleteAccount: async function() {
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
    }
};