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
                        <button class="btn btn-sm btn-outline-secondary edit-post" data-id="${post.id}" title="Edit">
                            <i class="bi bi-pencil"></i>
                        </button>
                        <button class="btn btn-sm btn-outline-danger delete-post" data-id="${post.id}" data-title="${post.title}" title="Delete">
                            <i class="bi bi-trash"></i>
                        </button>
                    </div>
                </div>
            </div>
        </div>
    `;
    
    // Add event listeners
    card.querySelector('.view-post').addEventListener('click', () => viewBlogPost(post.id));
    card.querySelector('.edit-post').addEventListener('click', () => editBlogPost(post.id));
    card.querySelector('.delete-post').addEventListener('click', (e) => {
        e.stopPropagation();
        deleteBlogPost(post.id, post.title);
    });
    
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
 */
function editBlogPost(postId) {
    // Implement edit functionality
    window.location.href = `edit-blog-post.html?id=${postId}`;
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
