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

// Global reference to the initialization function
window.initializeBlogManagement = async function() {
    console.log('Initializing blog management...');
    
    try {
        // Wait for the DOM to be fully loaded
        if (document.readyState === 'loading') {
            await new Promise(resolve => document.addEventListener('DOMContentLoaded', resolve));
        }
        
        // Wait for the API to be fully loaded
        await waitForAPI();
        
        // Initialize the page
        await initPage();
        
    } catch (error) {
        console.error('Error initializing blog management:', error);
        // Show error to user
        Swal.fire({
            icon: 'error',
            title: 'Initialization Error',
            text: 'Failed to initialize the blog management system. Please refresh the page and try again.',
            confirmButtonText: 'OK'
        });
    }
};

// Initialize the application
async function initializeApp() {
    console.log('Initializing blog management...');
    
    try {
        // Wait for the DOM to be fully loaded
        if (document.readyState === 'loading') {
            await new Promise(resolve => document.addEventListener('DOMContentLoaded', resolve));
        }
        
        // Wait for the API to be fully loaded
        await waitForAPI();
        
        // Check if user is logged in
        const user = window.api.auth.getCurrentUser();
        if (!user || !user.id) {
            console.log('User not authenticated, redirecting to login...');
            window.location.href = '/login.html';
            return;
        }
        
        // Initialize the page
        await initPage();
        await loadUserBlogPosts();

        // Event listeners are set up in individual components
        console.log('Application initialized successfully');
    } catch (error) {
        console.error('Initialization error:', error);
        
        // Show user-friendly error message
        if (typeof Swal !== 'undefined') {
            Swal.fire({
                icon: 'error',
                title: 'Initialization Error',
                text: error.message || 'Failed to initialize the application. Please refresh the page.',
                confirmButtonText: 'Refresh',
                allowOutsideClick: false
            }).then(() => {
                window.location.reload();
            });
        } else {
            // Fallback to simple alert if Swal is not available
            const errorContainer = document.getElementById('errorContainer');
            if (errorContainer) {
                errorContainer.textContent = 'Failed to initialize application. Please refresh the page.';
                errorContainer.classList.remove('d-none');
            } else {
                alert('Failed to initialize application. Please refresh the page.');
            }
        }
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
            // Store current URL to redirect back after login
            const returnUrl = encodeURIComponent(window.location.pathname + window.location.search);
            window.location.href = `/login.html?returnUrl=${returnUrl}`;
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
            
            if (response.status === 401) {
                // Token expired or invalid, log out and redirect to login
                console.log('Authentication required, redirecting to login...');
                if (window.api?.auth?.logout) {
                    window.api.auth.logout();
                }
                const returnUrl = encodeURIComponent(window.location.pathname + window.location.search);
                window.location.href = `/login.html?returnUrl=${returnUrl}`;
                return;
            }
            
            if (!response.ok) {
                let errorMessage = 'Failed to load blog posts';
                try {
                    const errorData = await response.json();
                    errorMessage = errorData.msg || errorData.error || errorMessage;
                    console.error('API Error:', errorData);
                    
                    // Handle token expiration specifically
                    if (errorData.msg === 'Token has expired' || errorData.error === 'Token has expired') {
                        console.log('Token expired, logging out and redirecting to login...');
                        if (window.api?.auth?.logout) {
                            window.api.auth.logout();
                        }
                        const returnUrl = encodeURIComponent(window.location.pathname + window.location.search);
                        window.location.href = `/login.html?returnUrl=${returnUrl}`;
                        return;
                    }
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
 * View a blog post in a modal
 * @param {number} postId - The ID of the post to view
 */
async function viewBlogPost(postId) {
    const modal = new bootstrap.Modal(document.getElementById('viewBlogPostModal'));
    const loadingElement = document.getElementById('viewPostLoading');
    const contentElement = document.getElementById('viewPostContent');
    const errorElement = document.getElementById('viewPostError');
    
    try {
        // Show loading state
        loadingElement.classList.remove('d-none');
        contentElement.classList.add('d-none');
        errorElement.classList.add('d-none');
        
        // Show the modal
        modal.show();
        
        // Fetch the blog post
        const post = await window.api.blog.getPost(postId);
        
        if (!post) {
            throw new Error('Blog post not found');
        }
        
        // Populate the modal with post data
        populateViewModal(post);
        
        // Set up the edit button to open the edit modal
        const editBtn = document.getElementById('editPostBtn');
        if (editBtn) {
            editBtn.onclick = () => {
                modal.hide();
                editBlogPost(postId);
            };
        }
        
        // Show the content
        loadingElement.classList.add('d-none');
        contentElement.classList.remove('d-none');
        
    } catch (error) {
        console.error('Error loading blog post:', error);
        loadingElement.classList.add('d-none');
        errorElement.classList.remove('d-none');
        errorElement.textContent = error.message || 'Failed to load the blog post. Please try again.';
    }
}

/**
 * Populate the view modal with blog post data
 * @param {Object} post - The blog post data
 */
function populateViewModal(post) {
    // Set the title
    document.getElementById('viewPostTitle').textContent = post.title || 'Untitled Post';
    
    // Set the image
    const imageElement = document.getElementById('viewPostImage');
    if (post.image_url) {
        imageElement.src = post.image_url;
        imageElement.style.display = 'block';
    } else {
        imageElement.style.display = 'none';
    }
    
    // Set the date
    const dateElement = document.getElementById('viewPostDate');
    if (post.date) {
        const date = new Date(post.date);
        dateElement.textContent = date.toLocaleDateString('en-US', {
            year: 'numeric',
            month: 'long',
            day: 'numeric'
        });
    } else {
        dateElement.textContent = '';
    }
    
    // Set the location (country and region)
    const locationElement = document.getElementById('viewPostLocation');
    if (locationElement) {
        const locationParts = [];
        if (post.country) locationParts.push(post.country);
        if (post.region && post.region !== post.country) locationParts.push(post.region);
        
        if (locationParts.length > 0) {
            locationElement.innerHTML = `<i class="fas fa-map-marker-alt me-1"></i> ${locationParts.join(', ')}`;
            locationElement.style.display = 'inline-block';
        } else {
            locationElement.style.display = 'none';
        }
    }
    
    // Set the author
    const authorElement = document.getElementById('viewPostAuthor');
    if (authorElement) {
        if (post.author) {
            authorElement.textContent = `By ${post.author}`;
        } else {
            authorElement.textContent = '';
        }
    }
    
    // Set the content - handle both HTML and plain text content
    const contentElement = document.getElementById('viewPostContentBody');
    if (!contentElement) {
        console.error('Content element not found');
        return;
    }
    
    if (!post.content) {
        contentElement.innerHTML = '<p class="text-muted">No content available.</p>';
        return;
    }

    // Function to clean and sanitize HTML content
    const cleanAndSanitizeHtml = (html) => {
        if (!html) return '';
        
        // Create a temporary div to parse the HTML
        const temp = document.createElement('div');
        temp.innerHTML = html;
        
        // Remove any script tags and other potentially dangerous elements
        const scripts = temp.getElementsByTagName('script');
        while (scripts[0]) {
            scripts[0].parentNode.removeChild(scripts[0]);
        }
        
        // Remove any style tags and links
        const styles = temp.getElementsByTagName('style');
        while (styles[0]) {
            styles[0].parentNode.removeChild(styles[0]);
        }
        
        const links = temp.getElementsByTagName('link');
        while (links[0]) {
            links[0].parentNode.removeChild(links[0]);
        }
        
        // Define allowed HTML tags
        const allowedTags = ['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'strong', 'em', 'u', 's', 'blockquote', 
                            'ul', 'ol', 'li', 'a', 'img', 'br', 'hr', 'div', 'span', 'pre', 'code', 'table', 
                            'thead', 'tbody', 'tr', 'th', 'td'];
        
        // Define allowed attributes for specific tags
        const allowedAttributes = {
            'a': ['href', 'title', 'target'],
            'img': ['src', 'alt', 'title', 'width', 'height', 'class'],
            'code': ['class'],
            'table': ['class', 'border', 'cellspacing', 'cellpadding'],
            'th': ['scope', 'colspan', 'rowspan'],
            'td': ['colspan', 'rowspan']
        };
        
        // Process all elements in the content
        const processNode = (node) => {
            // Process child nodes first (depth-first)
            for (let i = node.childNodes.length - 1; i >= 0; i--) {
                const child = node.childNodes[i];
                
                // Remove comments
                if (child.nodeType === Node.COMMENT_NODE) {
                    node.removeChild(child);
                    continue;
                }
                
                // Process element nodes
                if (child.nodeType === Node.ELEMENT_NODE) {
                    const tagName = child.tagName.toLowerCase();
                    
                    // Remove disallowed tags but keep their content
                    if (!allowedTags.includes(tagName)) {
                        const fragment = document.createDocumentFragment();
                        while (child.firstChild) {
                            fragment.appendChild(child.firstChild);
                        }
                        node.replaceChild(fragment, child);
                        continue;
                    }
                    
                    // Process attributes
                    const allowedAttrs = allowedAttributes[tagName] || [];
                    const attrs = child.attributes;
                    for (let i = attrs.length - 1; i >= 0; i--) {
                        const attr = attrs[i];
                        const attrName = attr.name.toLowerCase();
                        
                        // Remove disallowed attributes
                        if (!allowedAttrs.includes(attrName)) {
                            child.removeAttribute(attr.name);
                            continue;
                        }
                        
                        // Sanitize specific attributes
                        if (attrName === 'href' || attrName === 'src') {
                            // Only allow http, https, and relative URLs
                            const url = attr.value.trim();
                            if (!/^(https?:\/\/|\/|#)/i.test(url)) {
                                child.removeAttribute(attr.name);
                            } else if (url.startsWith('javascript:')) {
                                child.removeAttribute(attr.name);
                            }
                        }
                    }
                    
                    // Process child nodes
                    processNode(child);
                }
            }
        };
        
        // Start processing from the root
        processNode(temp);
        
        // Return the sanitized HTML
        return temp.innerHTML || 'No content available.';
    };
    
    // Set the cleaned and sanitized content
    const cleanedContent = cleanAndSanitizeHtml(post.content);
    contentElement.innerHTML = cleanedContent;
    
    // Initialize any plugins or components within the blog content
    initializeBlogContent(contentElement, post);
}

/**
 * Initialize any plugins or components within the blog content
 * @param {HTMLElement} container - The container element to initialize components in
 * @param {Object} post - The blog post data (optional)
 */
function initializeBlogContent(container, post) {
    if (!container) return;
    
    try {
        // Initialize any tooltips within the content
        const tooltipTriggerList = [].slice.call(container.querySelectorAll('[data-bs-toggle="tooltip"]'));
        tooltipTriggerList.forEach(function (tooltipTriggerEl) {
            try {
                return new bootstrap.Tooltip(tooltipTriggerEl);
            } catch (error) {
                console.error('Error initializing tooltip:', error);
            }
        });
        
        // Initialize any popovers within the content
        const popoverTriggerList = [].slice.call(container.querySelectorAll('[data-bs-toggle="popover"]'));
        popoverTriggerList.forEach(function (popoverTriggerEl) {
            try {
                return new bootstrap.Popover(popoverTriggerEl);
            } catch (error) {
                console.error('Error initializing popover:', error);
            }
        });
        
        // Process any code blocks with syntax highlighting if post content has them
        const codeBlocks = container.querySelectorAll('pre code');
        if (codeBlocks.length > 0 && typeof hljs !== 'undefined') {
            codeBlocks.forEach((block) => {
                try {
                    hljs.highlightElement(block);
                } catch (error) {
                    console.error('Error highlighting code block:', error);
                }
            });
        }
        
        // Process any images to make them responsive
        const images = container.querySelectorAll('img');
        images.forEach((img) => {
            if (!img.classList.contains('img-fluid')) {
                img.classList.add('img-fluid');
            }
        });
        
        // Set the tags if post data is provided
        if (post) {
            const tagsElement = document.getElementById('viewPostTags');
            if (tagsElement) {
                if (post.tags) {
                    // If tags is a string, split it by comma, otherwise use as is
                    const tagsArray = typeof post.tags === 'string' 
                        ? post.tags.split(',').map(tag => tag.trim()).filter(tag => tag) 
                        : (Array.isArray(post.tags) ? post.tags : []);
                        
                    if (tagsArray.length > 0) {
                        tagsElement.innerHTML = tagsArray.map(tag => 
                            `<span class="badge bg-secondary me-1">${escapeHtml(tag)}</span>`
                        ).join('');
                    } else {
                        tagsElement.innerHTML = '<span class="text-muted">No tags</span>';
                    }
                } else {
                    tagsElement.innerHTML = '<span class="text-muted">No tags</span>';
                }
            }
        }
        
    } catch (error) {
        console.error('Error initializing blog content:', error);
    }
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
                <div id="editor-container" style="height: 300px;">${post.content || ''}</div>
                <textarea id="editPostContent" class="d-none" required></textarea>
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
                           value="${escapeHtml(post.country || '')}" 
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
    const editorContainer = document.getElementById('editor-container');
    const contentInput = document.getElementById('editPostContent');
    
    if (!editForm || !imageInput || !imagePreview || !removeImageBtn || !saveChangesBtn || !editorContainer || !contentInput) {
        console.error('Required form elements not found');
        return;
    }
    
    // Initialize Quill editor
    const quill = new Quill('#editor-container', {
        theme: 'snow',
        modules: {
            toolbar: [
                [{ 'header': [1, 2, 3, 4, 5, 6, false] }],
                ['bold', 'italic', 'underline', 'strike'],
                [{ 'list': 'ordered'}, { 'list': 'bullet' }],
                ['link', 'image'],
                ['clean']
            ]
        },
        placeholder: 'Write your blog post here...',
    });
    
    // Set the initial content
    if (post.content) {
        quill.clipboard.dangerouslyPasteHTML(post.content);
    }
    
    // Update the hidden textarea with the HTML content when the form is submitted
    editForm.onsubmit = function(e) {
        e.preventDefault();
        contentInput.value = quill.root.innerHTML;
        // Handle form submission here
    };
    
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
    
    // Get the Quill editor instance
    const quill = Quill.find(document.querySelector('#editor-container'));
    if (!quill) {
        console.error('Quill editor not found');
        return;
    }
    
    const postId = form.dataset.postId;
    const title = document.getElementById('editPostTitle')?.value.trim();
    const content = quill.root.innerHTML; // Get HTML content from Quill
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
                console.log('Uploading file:', file.name, 'Size:', file.size, 'bytes');
                const uploadResponse = await window.api.blog.uploadImage(file);
                console.log('Upload response:', uploadResponse);
                
                // Make sure we have a valid URL (handle both response formats)
                imageUrl = uploadResponse.url || uploadResponse.imageUrl;
                if (!imageUrl) {
                    throw new Error('No URL returned from server');
                }
                
                console.log('Image uploaded successfully. URL:', imageUrl);
                
                // Update the preview immediately
                const preview = document.getElementById('editPostImagePreview');
                if (preview) {
                    preview.src = imageUrl;
                    preview.classList.remove('d-none');
                }
                
                // Enable the remove image button if it exists
                if (removeImageBtn) {
                    removeImageBtn.disabled = false;
                }
                
            } catch (error) {
                console.error('Error uploading image:', error);
                throw new Error('Failed to upload image. Please try again.');
            }
        } else if (removeImageBtn && removeImageBtn.disabled) {
            // Image was removed
            console.log('Image was removed');
            imageUrl = '';
            
            // Clear the preview
            const preview = document.getElementById('editPostImagePreview');
            if (preview) {
                preview.src = '#';
                preview.classList.add('d-none');
            }
        }
        
        // Process tags - ensure it's a string of comma-separated values
        let processedTags = '';
        if (tags) {
            if (Array.isArray(tags)) {
                processedTags = tags.join(',');
            } else if (typeof tags === 'string') {
                processedTags = tags.split(',').map(tag => tag.trim()).filter(tag => tag).join(',');
            }
        }

        // Prepare the update data
        const updateData = {
            title,
            content,
            region: region || null,
            country: country || null,
            tags: processedTags
        };
        
        // Only include image_url if we have one
        if (imageUrl) {
            updateData.image_url = imageUrl;
        }
        
        // Update the blog post
        const response = await window.api.blog.updatePost(postId, updateData);
        
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
