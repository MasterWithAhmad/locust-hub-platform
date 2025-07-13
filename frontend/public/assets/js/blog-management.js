/**
 * Blog Management Script
 * Handles all blog-related functionality including CRUD operations
 */

/**
 * Shows a toast notification
 * @param {string} message - The message to display
 * @param {string} type - Type of toast (success, error, warning, info)
 * @param {number} [duration=3000] - How long to show the toast in milliseconds
 */
function showToast(message, type = 'info', duration = 3000) {
    const toastContainer = document.getElementById('toastContainer') || document.body;
    const toastId = 'toast-' + Date.now();
    const iconClass = {
        success: 'bi-check-circle',
        error: 'bi-exclamation-triangle',
        warning: 'bi-exclamation-circle',
        info: 'bi-info-circle'
    }[type] || 'bi-info-circle';

    const bgClass = {
        success: 'bg-success',
        error: 'bg-danger',
        warning: 'bg-warning',
        info: 'bg-primary'
    }[type] || 'bg-primary';

    const toastHtml = `
        <div id="${toastId}" class="toast align-items-center text-white ${bgClass} border-0" 
             role="alert" aria-live="assertive" aria-atomic="true">
            <div class="d-flex">
                <div class="toast-body">
                    <i class="bi ${iconClass} me-2"></i>
                    ${message}
                </div>
                <button type="button" class="btn-close btn-close-white me-2 m-auto" 
                        data-bs-dismiss="toast" aria-label="Close"></button>
            </div>
        </div>
    `;
    
    const toastElement = document.createElement('div');
    toastElement.innerHTML = toastHtml;
    const toastNode = toastElement.firstElementChild;
    toastContainer.appendChild(toastNode);
    
    const toast = new bootstrap.Toast(toastNode, {
        autohide: true,
        delay: duration
    });
    toast.show();
    
    // Clean up after toast is hidden
    toastNode.addEventListener('hidden.bs.toast', function() {
        toastNode.remove();
    });
}

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
    
    // Initialize image upload functionality
    initializeImageUpload();
    
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
async function initPage() {
    try {
        console.log('Initializing blog management page...');
        
        // Check if we're on the blog management page by looking for the blog posts grid
        const blogPostsGrid = document.getElementById('blogPostsGrid');
        if (!blogPostsGrid) {
            console.log('Blog posts grid not found, not on blog management page');
            return;
        }
        
        // Load user data
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
            
            // Load blog posts
            console.log('Loading blog posts...');
            await loadUserBlogPosts();
        } else {
            console.log('No user logged in, redirecting to login...');
            const returnUrl = encodeURIComponent(window.location.pathname + window.location.search);
            window.location.href = `/login.html?returnUrl=${returnUrl}`;
        }
    } catch (error) {
        console.error('Error initializing page:', error);
        // Show error to user
        showToast('Failed to initialize the blog management system. Please refresh the page and try again.', 'error', 5000);
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
    console.log('loadUserBlogPosts function called');
    
    const blogPostsGrid = document.getElementById('blogPostsGrid');
    const noPostsMessage = document.getElementById('noPostsMessage');
    
    console.log('Blog posts grid element:', blogPostsGrid);
    console.log('No posts message element:', noPostsMessage);
    
    // Make sure elements exist before proceeding
    if (!blogPostsGrid || !noPostsMessage) {
        console.error('Required DOM elements not found');
        return;
    }
    
    try {
        console.log('Starting to load user blog posts...');
        
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
            const apiUrl = '/api/users/me/blogposts';
            const authHeader = getAuthHeader();
            
            console.log('Making API call to:', apiUrl);
            console.log('Auth headers:', authHeader);
            
            const response = await fetch(apiUrl, {
                method: 'GET',
                headers: authHeader,
                credentials: 'include'
            });

            console.log('Response status:', response.status);
            console.log('Response headers:', Object.fromEntries([...response.headers.entries()]));
            
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

            const responseData = await response.json();
            console.log('Raw API response data:', responseData);
            
            // Handle case where response might be an object with a data property
            const posts = Array.isArray(responseData) ? responseData : (responseData.data || []);
            console.log('Processed posts data:', posts);
            
            // Clear loading state
            blogPostsGrid.innerHTML = '';
            
            if (!Array.isArray(posts) || posts.length === 0) {
                console.log('No blog posts found in the response');
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
        const defaultImage = '/assets/blog_images/blog_c958fcef91374d64ad27ea17feebdce2.webp';
        
        if (post.image_url) {
            let imageUrl = post.image_url;
            
            // Handle different URL formats
            if (!imageUrl.startsWith('http') && !imageUrl.startsWith('/')) {
                // If it's just a filename, construct the full URL
                imageUrl = `/api/uploads/blog_images/${imageUrl}`;
            }
            
            imagePreview.src = imageUrl;
            imagePreview.onerror = function() {
                this.src = defaultImage; // Fallback to local default image
                this.onerror = null; // Prevent infinite loop if default image also fails
            };
            
            if (removeImageBtn) {
                removeImageBtn.disabled = false;
            }
        } else {
            imagePreview.src = defaultImage; // Use local default image
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
// Global variable to store the selected image file
let selectedImageFile = null;

/**
 * Initialize image upload functionality
 */
function initializeImageUpload() {
    const imageInput = document.getElementById('editPostImage');
    const imagePreview = document.getElementById('editPostImagePreview');
    const removeImageBtn = document.getElementById('removeImageBtn');

    if (!imageInput || !imagePreview) {
        console.error('Required image upload elements not found');
        return;
    }

    // Set default image if none is set
    if (!imagePreview.src || imagePreview.src.includes('placeholder')) {
        imagePreview.src = '/assets/img/placeholder-blog.jpg'; // Use a local placeholder
    }

    // Handle file selection
    const handleFileSelect = (e) => {
        const file = e?.target?.files?.[0];
        if (!file) return;

        // Validate file type
        if (!file.type.match('image.*')) {
            showToast('Please select a valid image file (JPEG, PNG, etc.)', 'error', 3000);
            return;
        }

        // Validate file size (5MB max)
        const maxSize = 5 * 1024 * 1024; // 5MB
        if (file.size > maxSize) {
            showToast('Image size should be less than 5MB', 'error', 3000);
            return;
        }

        // Create preview
        const reader = new FileReader();
        reader.onload = (e) => {
            try {
                imagePreview.src = e.target.result;
                selectedImageFile = file;
                if (removeImageBtn) removeImageBtn.disabled = false;
            } catch (error) {
                console.error('Error creating image preview:', error);
                showToast('Error processing image', 'error', 3000);
            }
        };
        reader.onerror = () => {
            console.error('Error reading file');
            showToast('Error reading image file', 'error', 3000);
        };
        reader.readAsDataURL(file);
    };

    // Handle remove image
    const handleRemoveImage = () => {
        try {
            if (imageInput) imageInput.value = '';
            imagePreview.src = '/assets/img/placeholder-blog.jpg';
            selectedImageFile = null;
            if (removeImageBtn) removeImageBtn.disabled = true;
        } catch (error) {
            console.error('Error removing image:', error);
        }
    };

    // Add event listeners
    imageInput.addEventListener('change', handleFileSelect);
    if (removeImageBtn) {
        removeImageBtn.addEventListener('click', handleRemoveImage);
    }

    // Cleanup function
    return () => {
        imageInput.removeEventListener('change', handleFileSelect);
        if (removeImageBtn) {
            removeImageBtn.removeEventListener('click', handleRemoveImage);
        }
    };
}

/**
 * Handle edit form submission
 * @param {Event} e - The form submission event
 */
async function handleEditFormSubmit(e) {
    e.preventDefault();
    
    const form = document.getElementById('editBlogPostForm');
    if (!form) {
        console.error('Edit form not found');
        return;
    }
    
    const postId = form.dataset.postId;
    const title = document.getElementById('editPostTitle')?.value.trim();
    const content = document.getElementById('editPostContent')?.value.trim();
    const region = document.getElementById('editPostRegion')?.value.trim();
    const country = document.getElementById('editPostCountry')?.value.trim();
    const tags = document.getElementById('editPostTags')?.value.trim();
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
        let imageUrl = undefined;
        
        // Handle image upload if a new file was selected
        if (selectedImageFile) {
            try {
                // Show loading state
                if (saveBtn && saveBtnSpinner && saveBtnText) {
                    saveBtn.disabled = true;
                    saveBtnSpinner.classList.remove('d-none');
                    saveBtnText.textContent = ' Uploading Image...';
                }

                // Create FormData for the file upload
                const formData = new FormData();
                formData.append('image', selectedImageFile);
                
                // Upload the image
                const uploadResponse = await window.api.blog.uploadImage(selectedImageFile);
                
                if (uploadResponse && (uploadResponse.url || uploadResponse.imageUrl)) {
                    imageUrl = uploadResponse.url || uploadResponse.imageUrl;
                    console.log('Image uploaded successfully:', imageUrl);
                } else {
                    throw new Error('Invalid response from server');
                }
            } catch (error) {
                console.error('Error uploading image:', error);
                showToast(error.message || 'Failed to upload image. Please try again.', 'error', 5000);
                
                // Reset button state on error
                if (saveBtn && saveBtnSpinner && saveBtnText) {
                    saveBtn.disabled = false;
                    saveBtnSpinner.classList.add('d-none');
                    saveBtnText.textContent = 'Save Changes';
                }
                return;
            }
        } else if (removeImageBtn && removeImageBtn.disabled) {
            // Image was removed
            imageUrl = '';
            console.log('Image was removed from the post');
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
        await window.api.blog.updatePost(postId, postData);
        
        // Get the modal element before doing anything else
        const modalElement = document.getElementById('editBlogPostModal');
        const modal = bootstrap.Modal.getInstance(modalElement);
        
        // Reset the form and image preview
        if (form) form.reset();
        const imagePreview = document.getElementById('editPostImagePreview');
        if (imagePreview) {
            imagePreview.src = '/assets/img/placeholder-blog.jpg';
        }
        if (removeImageBtn) {
            removeImageBtn.disabled = true;
        }
        selectedImageFile = null;
        window.newImageFile = null;
        
        // Show success toast
        showToast('Blog post updated successfully!', 'success', 2000);
        
        // Close the modal
        if (modal) {
            // Listen for the hidden event to ensure the modal is fully hidden before reloading
            const handleHidden = () => {
                modalElement.removeEventListener('hidden.bs.modal', handleHidden);
                // Reload the posts after the modal is fully hidden
                loadUserBlogPosts();
            };
            
            modalElement.addEventListener('hidden.bs.modal', handleHidden);
            modal.hide();
        } else {
            // If we can't get the modal instance, just reload immediately
            loadUserBlogPosts();
        }
        
    } catch (error) {
        console.error('Error updating blog post:', error);
        showToast(error.message || 'Failed to update blog post. Please try again.', 'error', 5000);
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
    // Create a confirmation message
    const message = `Are you sure you want to delete the post "${postTitle}"? This action cannot be undone.`;
    
    // Create a container for the confirmation dialog
    const container = document.createElement('div');
    container.className = 'delete-confirmation-dialog';
    container.innerHTML = `
        <div class="modal fade" id="deleteConfirmationModal" tabindex="-1" aria-labelledby="deleteConfirmationModalLabel" aria-hidden="true">
            <div class="modal-dialog">
                <div class="modal-content">
                    <div class="modal-header">
                        <h5 class="modal-title" id="deleteConfirmationModalLabel">Confirm Deletion</h5>
                        <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
                    </div>
                    <div class="modal-body">
                        <p>${message}</p>
                    </div>
                    <div class="modal-footer">
                        <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Cancel</button>
                        <button type="button" class="btn btn-danger" id="confirmDeleteBtn">Delete</button>
                    </div>
                </div>
            </div>
        </div>
    `;
    
    // Add the modal to the document
    document.body.appendChild(container);
    
    // Show the modal
    const modal = new bootstrap.Modal(container.querySelector('#deleteConfirmationModal'));
    modal.show();
    
    // Handle the delete button click
    const confirmBtn = container.querySelector('#confirmDeleteBtn');
    confirmBtn.addEventListener('click', async () => {
        try {
            // Show loading state
            confirmBtn.disabled = true;
            confirmBtn.innerHTML = '<span class="spinner-border spinner-border-sm me-1" role="status" aria-hidden="true"></span> Deleting...';
            
            // Call the API to delete the post
            await window.api.blog.deletePost(postId);
            
            // Show success message
            showToast('Blog post deleted successfully!', 'success', 3000);
            
            // Close the modal
            modal.hide();
            
            // Remove the modal from the DOM
            container.remove();
            
            // Reload the posts
            loadUserBlogPosts();
            
        } catch (error) {
            console.error('Error deleting blog post:', error);
            showToast(error.message || 'Failed to delete blog post. Please try again.', 'error', 5000);
            
            // Reset the button state
            confirmBtn.disabled = false;
            confirmBtn.textContent = 'Delete';
        }
    });
    
    // Clean up the modal when it's closed
    container.querySelector('#deleteConfirmationModal').addEventListener('hidden.bs.modal', () => {
        container.remove();
    });
}

/**
 * Delete a blog post
 * @param {number} postId - The ID of the post to delete
 * @param {string} postTitle - The title of the post (for confirmation dialog)
 */
async function deleteBlogPost(postId, postTitle) {
    // Create a confirmation modal
    const modalId = 'deleteConfirmationModal';
    const modalHtml = `
        <div class="modal fade" id="${modalId}" tabindex="-1" aria-labelledby="${modalId}Label" aria-hidden="true">
            <div class="modal-dialog">
                <div class="modal-content">
                    <div class="modal-header">
                        <h5 class="modal-title" id="${modalId}Label">Delete Blog Post</h5>
                        <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
                    </div>
                    <div class="modal-body">
                        <p>Are you sure you want to delete <strong>${escapeHtml(postTitle)}</strong>?<br>This action cannot be undone.</p>
                    </div>
                    <div class="modal-footer">
                        <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Cancel</button>
                        <button type="button" class="btn btn-danger" id="confirmDeleteBtn">
                            <span class="spinner-border spinner-border-sm d-none" role="status" aria-hidden="true"></span>
                            <span class="btn-text">Delete</span>
                        </button>
                    </div>
                </div>
            </div>
        </div>
    `;

    // Add modal to the document
    const modalContainer = document.createElement('div');
    modalContainer.innerHTML = modalHtml;
    document.body.appendChild(modalContainer);

    // Show the modal
    const modalElement = document.getElementById(modalId);
    const modal = new bootstrap.Modal(modalElement);
    modal.show();

    // Handle delete button click
    const confirmBtn = document.getElementById('confirmDeleteBtn');
    const spinner = confirmBtn.querySelector('.spinner-border');
    const btnText = confirmBtn.querySelector('.btn-text');

    return new Promise((resolve) => {
        confirmBtn.addEventListener('click', async () => {
            try {
                // Show loading state
                confirmBtn.disabled = true;
                spinner.classList.remove('d-none');
                btnText.textContent = 'Deleting...';

                console.log('Deleting post with ID:', postId);
                const response = await window.api.blog.deletePost(postId);
                console.log('Delete successful:', response);
                
                // Show success message
                const successMessage = response?.message || 'The blog post has been deleted.';
                showToast(successMessage, 'success', 2000);

                // Close the modal
                modal.hide();
                
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
                
                resolve(true);
            } catch (error) {
                console.error('Delete error:', error);
                showToast(error.message || 'Failed to delete blog post', 'error', 5000);
                
                // Reset button state
                confirmBtn.disabled = false;
                spinner.classList.add('d-none');
                btnText.textContent = 'Delete';
                
                resolve(false);
            }
        });
        
        // Clean up the modal when it's closed
        modalElement.addEventListener('hidden.bs.modal', () => {
            modal.dispose();
            modalContainer.remove();
            resolve(false);
        });
    });
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
