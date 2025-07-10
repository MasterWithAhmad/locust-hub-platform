/**
 * Blog Management Script
 * Handles all blog-related functionality including CRUD operations
 */

// Wait for the API to be fully loaded with a timeout
function waitForAPI() {
    return new Promise((resolve, reject) => {
        const maxAttempts = 30; // 3 seconds total (30 * 100ms)
        let attempts = 0;
        
        const checkAPI = setInterval(() => {
            attempts++;
            
            if (window.api) {
                clearInterval(checkAPI);
                console.log('API loaded successfully');
                resolve();
            } else if (attempts >= maxAttempts) {
                clearInterval(checkAPI);
                console.error('Failed to load API after maximum attempts');
                reject(new Error('API initialization timed out. Please refresh the page.'));
            } else {
                console.log('Waiting for API to load...');
            }
        }, 100);
        
        // Initial check
        if (window.api) {
            clearInterval(checkAPI);
            resolve();
        }
    });
}

// Initialize the application
async function initializeApp() {
    try {
        console.log('Initializing application...');
        
        // Wait for API with timeout
        try {
            await waitForAPI();
        } catch (error) {
            console.error('API initialization error:', error);
            throw new Error('Failed to initialize application. Please refresh the page.');
        }
        
        // Check if user is logged in
        try {
            const user = window.api.auth.getCurrentUser();
            if (!user || !user.id) {
                console.log('User not authenticated, redirecting to login...');
                window.location.href = '/login.html';
                return;
            }

            // Initialize the page
            initPage();
            await loadUserBlogPosts();

            // Set up event listeners
            const createBtn = document.getElementById('createBlogPostBtn');
            if (createBtn) {
                createBtn.addEventListener('click', showCreateBlogPostModal);
            }
            
            console.log('Application initialized successfully');
        } catch (error) {
            console.error('Application initialization error:', error);
            throw new Error('Failed to load the application. Please try again.');
        }
    } catch (error) {
        console.error('Fatal initialization error:', error);
        // Show user-friendly error message
        Swal.fire({
            icon: 'error',
            title: 'Initialization Error',
            text: error.message || 'Failed to initialize the application. Please refresh the page.',
            confirmButtonText: 'Refresh',
            allowOutsideClick: false
        }).then(() => {
            window.location.reload();
        });
        
        // Re-throw to be caught by the global error handler
        throw error;
    }
}

/**
 * Initialize the page
 */
function initPage() {
    try {
        const user = window.api.auth.getCurrentUser();
        if (user) {
            // Update user info in the header
            const initials = user.full_name.split(' ').map(n => n[0]).join('').toUpperCase();
            const userInitials = document.getElementById('userInitials');
            const userName = document.getElementById('userName');
            const userFullName = document.getElementById('userFullName');
            
            if (userInitials) userInitials.textContent = initials;
            if (userName) userName.textContent = user.full_name;
            if (userFullName) userFullName.textContent = user.full_name;
        }
    } catch (error) {
        console.error('Error initializing page:', error);
    }
}

/**
 * Format date to a readable format
 */
function formatDate(dateString) {
    const options = { year: 'numeric', month: 'short', day: 'numeric' };
    return new Date(dateString).toLocaleDateString(undefined, options);
}

/**
 * Truncate text to a specified length
 */
function truncateText(text, maxLength = 100) {
    if (!text) return '';
    if (text.length <= maxLength) return text;
    return text.substring(0, maxLength) + '...';
}

/**
 * Create a blog post card element
 */
function createBlogPostCard(post) {
    const card = document.createElement('div');
    card.className = 'col-12 col-sm-6 col-md-4 col-lg-4 col-xxl-3 blog-card-column mb-4';
    card.setAttribute('data-post-id', post.id);
    
    // Clean and prepare content for display
    const cleanContent = post.content 
        ? post.content.replace(/<[^>]*>?/gm, '') // Remove HTML tags
        : '';
    
    card.innerHTML = `
        <div class="blog-card">
            <div class="blog-card-img-container">
                <img src="${post.image_url || 'https://via.placeholder.com/300x140?text=No+Image'}" 
                     alt="${post.title || ''}" 
                     class="blog-card-img"
                     onerror="this.src='https://via.placeholder.com/300x140?text=No+Image';">
            </div>
            <div class="blog-card-body">
                <div class="d-flex justify-content-end mb-2">
                    <small class="text-muted">${formatDate(post.date)}</small>
                </div>
                <h5 class="blog-card-title">${post.title || 'Untitled Post'}</h5>
                <p class="blog-card-text">${truncateText(cleanContent, 100)}</p>
                <div class="blog-card-actions">
                    <button class="btn btn-sm btn-outline-primary view-post" data-id="${post.id}">
                        <i class="bi bi-eye me-1"></i> View
                    </button>
                    <div class="btn-group">
                        <button class="btn btn-sm btn-outline-secondary edit-post" 
                                data-id="${post.id}" 
                                data-bs-toggle="tooltip" 
                                data-bs-placement="top" 
                                title="Edit Post">
                            <i class="bi bi-pencil"></i>
                        </button>
                        <button class="btn btn-sm btn-outline-danger delete-post" 
                                data-id="${post.id}" 
                                data-title="${post.title}" 
                                data-bs-toggle="tooltip" 
                                data-bs-placement="top" 
                                title="Delete Post">
                            <i class="bi bi-trash"></i>
                        </button>
                    </div>
                </div>
            </div>
        </div>
    `;
    
    // Add event listeners
    card.querySelector('.view-post').addEventListener('click', () => viewBlogPost(post.id));
    
    const editBtn = card.querySelector('.edit-post');
    editBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        editBlogPost(post.id);
    });
    
    const deleteBtn = card.querySelector('.delete-post');
    deleteBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        deleteBlogPost(post.id, post.title);
    });
    
    // Initialize tooltips
    if (typeof bootstrap !== 'undefined') {
        new bootstrap.Tooltip(editBtn);
        new bootstrap.Tooltip(deleteBtn);
    }
    
    return card;
}

/**
 * Load the current user's blog posts
 */
async function loadUserBlogPosts() {
    const blogPostsGrid = document.getElementById('blogPostsGrid');
    const noPostsMessage = document.getElementById('noPostsMessage');
    
    // Make sure elements exist before proceeding
    if (!blogPostsGrid || !noPostsMessage) {
        console.error('Required DOM elements not found');
        return;
    }
    
    try {
        console.log('Loading user blog posts...');
        
        // Show loading state
        blogPostsGrid.innerHTML = '<div class="col-12 text-center py-5"><div class="spinner-border text-primary" role="status"><span class="visually-hidden">Loading...</span></div></div>';
        
        // Get current user
        const user = window.api?.auth?.getCurrentUser?.();
        if (!user?.id) {
            console.warn('User not authenticated, redirecting to login...');
            window.location.href = '/login.html';
            return;
        }

        try {
            console.log('Fetching blog posts for current user...');
            const response = await fetch('/api/users/me/blogposts', {
                method: 'GET',
                headers: getAuthHeader(),
                credentials: 'include'
            });

            console.log('Response status:', response.status);
            
            if (!response.ok) {
                let errorMessage = 'Failed to load blog posts';
                try {
                    const errorData = await response.json();
                    errorMessage = errorData.error || errorMessage;
                    console.error('API Error:', errorData);
                } catch (e) {
                    const errorText = await response.text();
                    console.error('Error parsing error response:', e, 'Response:', errorText);
                    errorMessage = `${errorMessage} (Status: ${response.status})`;
                }
                throw new Error(errorMessage);
            }

            const posts = await response.json();
            console.log('Received posts:', posts);
            
            // Clear loading state
            blogPostsGrid.innerHTML = '';
            
            if (!Array.isArray(posts) || posts.length === 0) {
                console.log('No blog posts found');
                noPostsMessage.style.display = 'flex';
                return;
            }

            noPostsMessage.style.display = 'none';
            
            // Create and append blog post cards
            posts.forEach(post => {
                try {
                    const card = createBlogPostCard(post);
                    if (card) {
                        blogPostsGrid.appendChild(card);
                    }
                } catch (cardError) {
                    console.error('Error creating blog post card:', cardError);
                }
            });
            
            console.log('Successfully loaded', posts.length, 'blog posts');
            
        } catch (fetchError) {
            console.error('Error fetching blog posts:', fetchError);
            throw fetchError; // Re-throw to be caught by the outer catch
        }

    } catch (error) {
        console.error('Error in loadUserBlogPosts:', error);
        // Only update the DOM if the elements still exist
        if (blogPostsGrid && !document.body.contains(blogPostsGrid)) {
            console.warn('blogPostsGrid no longer in DOM, not updating UI');
            return;
        }
        
        blogPostsGrid.innerHTML = `
            <div class="col-12">
                <div class="alert alert-danger" role="alert">
                    <i class="bi bi-exclamation-triangle-fill me-2"></i>
                    ${error.message || 'Failed to load blog posts. Please try again later.'}
                </div>
            </div>`;
            
        // If it's an authentication error, redirect to login
        if (error.message.includes('authenticated') || error.message.includes('401')) {
            setTimeout(() => {
                window.location.href = '/login.html';
            }, 2000);
        }
    }
}

/**
 * Show the create blog post modal
 */
function showCreateBlogPostModal() {
    // You can implement a modal or redirect to a new page for creating a post
    // For now, we'll just show an alert
    Swal.fire({
        title: 'Create New Blog Post',
        html: `
            <div class="mb-3">
                <label for="postTitle" class="form-label">Title</label>
                <input type="text" class="form-control" id="postTitle" required>
            </div>
            <div class="mb-3">
                <label for="postContent" class="form-label">Content</label>
                <textarea class="form-control" id="postContent" rows="5" required></textarea>
            </div>
            <div class="row">
                <div class="col-md-6 mb-3">
                    <label for="postRegion" class="form-label">Region</label>
                    <input type="text" class="form-control" id="postRegion">
                </div>
                <div class="col-md-6 mb-3">
                    <label for="postCountry" class="form-label">Country</label>
                    <input type="text" class="form-control" id="postCountry">
                </div>
            </div>
            <div class="mb-3">
                <label for="postTags" class="form-label">Tags (comma-separated)</label>
                <input type="text" class="form-control" id="postTags" placeholder="e.g., locust, agriculture, prediction">
            </div>
        `,
        showCancelButton: true,
        confirmButtonText: 'Publish',
        cancelButtonText: 'Cancel',
        preConfirm: () => {
            return {
                title: document.getElementById('postTitle').value,
                content: document.getElementById('postContent').value,
                region: document.getElementById('postRegion').value,
                country: document.getElementById('postCountry').value,
                tags: document.getElementById('postTags').value
            };
        }
    }).then((result) => {
        if (result.isConfirmed && result.value) {
            createBlogPost(result.value);
        }
    });
}

/**
 * Create a new blog post
 */
async function createBlogPost(postData) {
    try {
        const user = api.auth.getCurrentUser();
        if (!user || !user.id) {
            throw new Error('User not found');
        }

        const response = await fetch(`${window.API_BASE_URL || 'http://localhost:5000/api'}/blogposts`, {
            method: 'POST',
            headers: getAuthHeader(),
            body: JSON.stringify({
                title: postData.title,
                content: postData.content,
                region: postData.region || null,
                country: postData.country || null,
                tags: postData.tags ? postData.tags.split(',').map(tag => tag.trim()) : [],
                user_id: user.id
            })
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.message || 'Failed to create blog post');
        }

        const newPost = await response.json();
        
        Swal.fire({
            icon: 'success',
            title: 'Success!',
            text: 'Your blog post has been published.',
            timer: 2000,
            showConfirmButton: false
        });

        // Reload the posts
        loadUserBlogPosts();

    } catch (error) {
        console.error('Error creating blog post:', error);
        Swal.fire({
            icon: 'error',
            title: 'Error',
            text: error.message || 'Failed to create blog post. Please try again.'
        });
    }
}

/**
 * View a blog post
 */
function viewBlogPost(postId) {
    window.location.href = `blog-post.html?id=${postId}`;
}

/**
 * Edit a blog post
 * @param {number} postId - The ID of the post to edit
 */
async function editBlogPost(postId) {
    const editModal = document.getElementById('editBlogPostModal');
    const formContent = document.getElementById('editFormContent');
    
    if (!editModal) {
        console.error('Edit modal not found');
        Swal.fire({
            icon: 'error',
            title: 'Error',
            text: 'Could not find the edit modal. Please refresh the page and try again.',
            confirmButtonText: 'OK'
        });
        return;
    }
    
    // Show loading state
    const showLoading = () => {
        if (formContent) {
            formContent.innerHTML = `
                <div class="text-center my-4">
                    <div class="spinner-border text-primary" role="status">
                        <span class="visually-hidden">Loading...</span>
                    </div>
                    <p class="mt-2">Loading post data...</p>
                </div>
            `;
        }
    };
    
    // Show error state
    const showError = (error) => {
        console.error('Error in editBlogPost:', error);
        
        // Extract error message from different possible error formats
        let errorMessage = 'Failed to load blog post for editing. Please try again.';
        if (error) {
            if (error.message) {
                errorMessage = error.message;
            } else if (typeof error === 'string') {
                errorMessage = error;
            } else if (error.response && error.response.error) {
                errorMessage = error.response.error;
            }
        }
        
        if (formContent) {
            formContent.innerHTML = `
                <div class="alert alert-danger">
                    <h5 class="alert-heading">Error</h5>
                    <p>${errorMessage}</p>
                    <div class="mt-3">
                        <button class="btn btn-secondary" onclick="this.closest('.modal').querySelector('.btn-close').click()">
                            Close
                        </button>
                        <button class="btn btn-primary ms-2" onclick="editBlogPost(${postId})">
                            <i class="bi bi-arrow-clockwise me-1"></i> Try Again
                        </button>
                    </div>
                </div>
            `;
        }
    };
    
    try {
        console.log(`Editing blog post with ID: ${postId}`);
        
        // Show the modal first
        const modal = new bootstrap.Modal(editModal);
        modal.show();
        
        // Set loading state
        showLoading();
        
        // Fetch the blog post data with a timeout
        const fetchPost = async () => {
            try {
                console.log('Fetching blog post data...');
                const post = await window.api.blog.getPost(postId);
                console.log('Received post data:', post);
                
                if (!post) {
                    throw new Error('No post data returned from the server');
                }
                
                // Populate the form with the post data
                populateEditForm(post);
                
            } catch (apiError) {
                console.error('API Error in fetchPost:', apiError);
                throw apiError;
            }
        };
        
        // Set a timeout for the API call
        const timeoutPromise = new Promise((_, reject) => {
            setTimeout(() => {
                reject(new Error('Request timed out. Please check your internet connection and try again.'));
            }, 10000); // 10 second timeout
        });
        
        // Race the API call against the timeout
        await Promise.race([fetchPost(), timeoutPromise]);
        
    } catch (error) {
        showError(error);
    }
}

/**
 * Populate the edit form with blog post data
 * @param {Object} post - The blog post data
 */
function populateEditForm(post) {
    const formContent = document.getElementById('editFormContent');
    const imagePreview = document.getElementById('editPostImagePreview');
    const removeImageBtn = document.getElementById('removeImageBtn');
    
    if (!formContent) return;
    
    // Format the date for the datetime-local input
    const postDate = new Date(post.date);
    const formattedDate = postDate.toISOString().slice(0, 16);
    
    // Create the edit form content (left side)
    formContent.innerHTML = `
        <form id="editBlogPostForm" data-post-id="${post.id}">
            <div class="mb-3">
                <label for="editPostTitle" class="form-label">Title <span class="text-danger">*</span></label>
                <input type="text" class="form-control" id="editPostTitle" value="${escapeHtml(post.title || '')}" required>
            </div>
            
            <div class="mb-3">
                <label for="editPostContent" class="form-label">Content <span class="text-danger">*</span></label>
                <textarea class="form-control" id="editPostContent" rows="10" required>${escapeHtml(post.content || '')}</textarea>
            </div>
            
            <div class="row g-3">
                <div class="col-md-6">
                    <label for="editPostRegion" class="form-label">Region</label>
                    <input type="text" class="form-control" id="editPostRegion" 
                           value="${escapeHtml(post.region || '')}" 
                           placeholder="e.g., North America">
                </div>
                <div class="col-md-6">
                    <label for="editPostCountry" class="form-label">Country</label>
                    <input type="text" class="form-control" id="editPostCountry" 
                           value="escapeHtml(post.country || '')}" 
                           placeholder="e.g., United States">
                </div>
            </div>
            
            <div class="mb-3">
                <label for="editPostTags" class="form-label mt-3">Tags</label>
                <input type="text" class="form-control" id="editPostTags" 
                       value="${escapeHtml(post.tags || '')}" 
                       placeholder="Comma-separated tags (e.g., agriculture, locust, forecast)">
                <div class="form-text">Separate tags with commas</div>
            </div>
        </form>
    `;
    
    // Update the image preview (right side)
    if (imagePreview) {
        if (post.image_url) {
            imagePreview.src = post.image_url;
            imagePreview.onerror = function() {
                this.src = 'https://via.placeholder.com/800x400?text=Image+Not+Found';
            };
            if (removeImageBtn) {
                removeImageBtn.disabled = false;
            }
        } else {
            imagePreview.src = 'https://via.placeholder.com/800x400?text=No+Image';
            if (removeImageBtn) {
                removeImageBtn.disabled = true;
            }
        }
    }
    
    // Initialize event listeners for the form
    setupEditFormEventListeners(post);
}

/**
 * Set up event listeners for the edit form
 * @param {Object} post - The blog post data
 */
function setupEditFormEventListeners(post) {
    const editForm = document.getElementById('editBlogPostForm');
    const imageInput = document.getElementById('editPostImage');
    const imagePreview = document.getElementById('editPostImagePreview');
    const removeImageBtn = document.getElementById('removeImageBtn');
    const saveChangesBtn = document.getElementById('saveChangesBtn');
    
    if (!editForm || !imageInput || !imagePreview || !removeImageBtn || !saveChangesBtn) {
        console.error('Required form elements not found');
        return;
    }
    
    // Handle image preview
    imageInput.addEventListener('change', function(e) {
        const file = e.target.files[0];
        if (file) {
            // Check file size (max 5MB)
            if (file.size > 5 * 1024 * 1024) {
                Swal.fire({
                    icon: 'error',
                    title: 'File too large',
                    text: 'Maximum file size is 5MB. Please choose a smaller image.'
                });
                this.value = ''; // Clear the file input
                return;
            }
            
            // Check file type
            if (!file.type.match('image.*')) {
                Swal.fire({
                    icon: 'error',
                    title: 'Invalid file type',
                    text: 'Please select a valid image file (JPEG, PNG, etc.)'
                });
                this.value = ''; // Clear the file input
                return;
            }
            
            // Create a preview URL for the selected image
            const reader = new FileReader();
            reader.onload = function(e) {
                imagePreview.src = e.target.result;
                removeImageBtn.disabled = false;
                removeImageBtn.dataset.imageRemoved = 'false';
            };
            reader.onerror = function() {
                console.error('Error reading image file');
                Swal.fire({
                    icon: 'error',
                    title: 'Error',
                    text: 'Failed to load the selected image. Please try again.'
                });
            };
            reader.readAsDataURL(file);
        }
    });
    
    // Handle remove image button
    removeImageBtn.addEventListener('click', function() {
        imagePreview.src = 'https://via.placeholder.com/800x400?text=No+Image';
        imageInput.value = ''; // Clear the file input
        removeImageBtn.disabled = true;
        
        // Set a flag to indicate the image was removed
        removeImageBtn.dataset.imageRemoved = 'true';
    });
    
    // Handle form submission
    editForm.addEventListener('submit', handleEditFormSubmit);
    
    // Also handle the save changes button click (in case the form doesn't submit properly)
    saveChangesBtn.addEventListener('click', function() {
        // Trigger the form submission
        const submitEvent = new Event('submit', {
            bubbles: true,
            cancelable: true
        });
        editForm.dispatchEvent(submitEvent);
    });
    
    // Add keyboard shortcut (Ctrl+Enter) to save the form
    document.addEventListener('keydown', function(e) {
        if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
            if (document.activeElement.closest('#editBlogPostModal')) {
                e.preventDefault();
                saveChangesBtn.click();
            }
        }
    });
}

/**
 * Handle edit form submission
 * @param {Event} e - The form submission event
 */
async function handleEditFormSubmit(e) {
    const form = document.getElementById('editBlogPostForm');
    if (!form) return;
    
    const postId = form.dataset.postId;
    const title = document.getElementById('editPostTitle')?.value.trim();
    const content = document.getElementById('editPostContent')?.value.trim();
    const region = document.getElementById('editPostRegion')?.value.trim();
    const country = document.getElementById('editPostCountry')?.value.trim();
    const tags = document.getElementById('editPostTags')?.value.trim();
    const imageInput = document.getElementById('editPostImage');
    const removeImageBtn = document.getElementById('removeImageBtn');
    
    // Validate required fields
    if (!title || !content) {
        Swal.fire({
            icon: 'error',
            title: 'Validation Error',
            text: 'Title and content are required.'
        });
        return;
    }
    
    // Show loading state
    const saveBtn = document.getElementById('saveChangesBtn');
    const saveBtnSpinner = saveBtn?.querySelector('.spinner-border');
    const saveBtnText = saveBtn?.querySelector('span:not(.spinner-border)');
    
    if (saveBtn && saveBtnSpinner && saveBtnText) {
        saveBtn.disabled = true;
        saveBtnSpinner.classList.remove('d-none');
        saveBtnText.textContent = ' Saving...';
    }
    
    try {
        let imageUrl = '';
        
        // Handle image upload if a new file was selected
        if (newImageFile) {
            try {
                const uploadResponse = await window.api.blog.uploadImage(newImageFile);
                imageUrl = uploadResponse.imageUrl;
            } catch (error) {
                console.error('Error uploading image:', error);
                throw new Error('Failed to upload image. Please try again.');
            }
        } else if (removeImageBtn && removeImageBtn.disabled) {
            // Image was removed
            imageUrl = '';
        }
        
        // Prepare the post data
        const postData = {
            title,
            content,
            region: region || null,
            country: country || null,
            tags: tags || null
        };
        
        // Only include image_url if it was changed
        if (imageUrl !== undefined) {
            postData.image_url = imageUrl;
        }
        
        // Update the blog post
        const response = await window.api.blog.updatePost(postId, postData);
        
        // Show success message
        Swal.fire({
            icon: 'success',
            title: 'Success',
            text: 'Blog post updated successfully!',
            timer: 2000,
            showConfirmButton: false
        });
        
        // Close the modal
        const modal = bootstrap.Modal.getInstance(document.getElementById('editBlogPostModal'));
        if (modal) modal.hide();
        
        // Reload the posts
        loadUserBlogPosts();
        
    } catch (error) {
        console.error('Error updating blog post:', error);
        
        // Show error message
        Swal.fire({
            icon: 'error',
            title: 'Error',
            text: error.message || 'Failed to update blog post. Please try again.'
        });
    } finally {
        // Reset button state
        if (saveBtn && saveBtnSpinner && saveBtnText) {
            saveBtn.disabled = false;
            saveBtnSpinner.classList.add('d-none');
            saveBtnText.textContent = 'Save Changes';
        }
    }
}


/**
 * Handle edit form submission
 * @param {Event} e - The form submission event
 */
async function handleEditFormSubmit(e) {
    e.preventDefault();
    
    const form = document.getElementById('editBlogPostForm');
    if (!form) return;
    
    const postId = form.dataset.postId;
    const title = document.getElementById('editPostTitle')?.value.trim();
    const content = document.getElementById('editPostContent')?.value.trim();
    const region = document.getElementById('editPostRegion')?.value.trim();
    const country = document.getElementById('editPostCountry')?.value.trim();
    const tags = document.getElementById('editPostTags')?.value.trim();
    const imageInput = document.getElementById('editPostImage');
    const removeImageBtn = document.getElementById('removeImageBtn');
    
    // Show loading state
    const saveBtn = document.getElementById('saveChangesBtn');
    const saveBtnSpinner = saveBtn?.querySelector('.spinner-border');
    const saveBtnText = saveBtn?.querySelector('span:not(.spinner-border)');
    
    if (saveBtn && saveBtnSpinner && saveBtnText) {
        saveBtn.disabled = true;
        saveBtnSpinner.classList.remove('d-none');
        saveBtnText.textContent = ' Saving...';
    }
    
    try {
        // Validate required fields
        if (!title || !content) {
            throw new Error('Title and content are required');
        }
        
        let imageUrl = null;
        
        // Handle image upload if a new file was selected
        const file = imageInput?.files[0];
        if (file) {
            try {
                const uploadResponse = await window.api.blog.uploadImage(file);
                imageUrl = uploadResponse.imageUrl;
            } catch (error) {
                console.error('Error uploading image:', error);
                throw new Error('Failed to upload image. Please try again.');
            }
        } else if (removeImageBtn && removeImageBtn.disabled) {
            // Image was removed
            imageUrl = '';
        }
        
        // Prepare the post data
        const postData = {
            title,
            content,
            region: region || null,
            country: country || null,
            tags: tags || null
        };
        
        // Only include image_url if it was changed
        if (imageUrl !== null) {
            postData.image_url = imageUrl;
        }
        
        // Update the blog post
        const response = await window.api.blog.updatePost(postId, postData);
        
        // Show success message
        Swal.fire({
            icon: 'success',
            title: 'Success',
            text: 'Blog post updated successfully!',
            timer: 2000,
            showConfirmButton: false
        });
        
        // Close the modal
        const modal = bootstrap.Modal.getInstance(document.getElementById('editBlogPostModal'));
        if (modal) modal.hide();
        
        // Reload the posts
        loadUserBlogPosts();
        
    } catch (error) {
        console.error('Error updating blog post:', error);
        
        // Show error message
        Swal.fire({
            icon: 'error',
            title: 'Error',
            text: error.message || 'Failed to update blog post. Please try again.'
        });
    } finally {
        // Reset button state
        if (saveBtn && saveBtnSpinner && saveBtnText) {
            saveBtn.disabled = false;
            saveBtnSpinner.classList.add('d-none');
            saveBtnText.textContent = 'Save Changes';
        }
    }
}

/**
 * Confirm blog post deletion
 * @param {number} postId - The ID of the post to delete
 * @param {string} postTitle - The title of the post (for confirmation)
 */
function confirmDeleteBlogPost(postId, postTitle) {
    Swal.fire({
        title: 'Delete Blog Post',
        html: `Are you sure you want to delete <strong>${escapeHtml(postTitle)}</strong>?<br>This action cannot be undone.`,
        icon: 'warning',
        showCancelButton: true,
        confirmButtonColor: '#d33',
        cancelButtonColor: '#6c757d',
        confirmButtonText: 'Yes, delete it!',
        cancelButtonText: 'Cancel',
        reverseButtons: true,
        focusCancel: true
    }).then((result) => {
        if (result.isConfirmed) {
            deleteBlogPost(postId, postTitle);
        }
    });
}

/**
 * Delete a blog post
 * @param {number} postId - The ID of the post to delete
 * @param {string} postTitle - The title of the post (for confirmation dialog)
 */
async function deleteBlogPost(postId, postTitle) {
    try {
        // Show confirmation dialog
        const result = await Swal.fire({
            title: 'Delete Blog Post',
            html: `Are you sure you want to delete <strong>${escapeHtml(postTitle)}</strong>?<n>This action cannot be undone.`,
            icon: 'warning',
            showCancelButton: true,
            confirmButtonColor: '#d33',
            cancelButtonColor: '#6c757d',
            confirmButtonText: 'Yes, delete it!',
            cancelButtonText: 'Cancel',
            reverseButtons: true,
            focusCancel: true,
            showLoaderOnConfirm: true,
            preConfirm: async () => {
                try {
                    console.log('Deleting post with ID:', postId);
                    const response = await window.api.blog.deletePost(postId);
                    console.log('Delete successful:', response);
                    return response;
                } catch (error) {
                    console.error('Delete error:', error);
                    Swal.showValidationMessage(
                        `Error: ${error.message || 'Failed to delete blog post'}`
                    );
                    return false;
                }
            },
            allowOutsideClick: () => !Swal.isLoading()
        });

        if (result.isConfirmed && result.value) {
            // Show success message
            const successMessage = result.value.message || 'The blog post has been deleted.';
            console.log('Deletion successful:', successMessage);
            
            await Swal.fire({
                icon: 'success',
                title: 'Deleted!',
                text: successMessage,
                timer: 2000,
                showConfirmButton: false
            });

            // Remove the deleted post card from the UI
            const deletedCard = document.querySelector(`[data-post-id="${postId}"]`);
            if (deletedCard) {
                deletedCard.style.opacity = '0';
                setTimeout(() => {
                    deletedCard.remove();
                    // Check if no posts are left
                    const blogPostsGrid = document.getElementById('blogPostsGrid');
                    if (blogPostsGrid && blogPostsGrid.children.length === 0) {
                        const noPostsMessage = document.getElementById('noPostsMessage');
                        if (noPostsMessage) {
                            noPostsMessage.style.display = 'flex';
                        }
                    }
                }, 300);
            } else {
                // If we can't find the specific card, reload all posts
                console.log('Card not found, reloading all posts');
                await loadUserBlogPosts();
            }
        }
    } catch (error) {
        console.error('Error in delete confirmation:', error);
        await Swal.fire({
            icon: 'error',
            title: 'Error',
            text: error.message || 'An unexpected error occurred. Please try again.',
            confirmButtonText: 'OK'
        });
    }
}

/**
 * Helper function to escape HTML
 */
function escapeHtml(unsafe) {
    if (!unsafe) return '';
    return unsafe
        .toString()
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#039;');
}
