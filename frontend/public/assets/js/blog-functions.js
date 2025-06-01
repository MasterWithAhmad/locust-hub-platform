/**
 * blog-functions.js
 * Handles all blog-specific functionality for the LocustHub blog page.
 * This includes:
 * - Fetching and displaying blog posts
 * - Search and filter functionality
 * - Pagination
 * - Opening individual blog posts in a modal
 * - Integration with the API for fetching blog data.
 */

// Global variables to store blog posts and current state
let allPosts = []; // Stores all fetched blog posts
let filteredPosts = []; // Stores posts after applying search/filters
let currentPage = 1; // Current page number for pagination
const postsPerPage = 6; // Number of posts to display per page

// API base URL - ensure this matches the backend API configuration
// It dynamically sets the API_BASE_URL based on the current origin
// to work correctly in both development (localhost:3000) and production environments.
const API_BASE_URL = window.location.origin.includes('3000') 
  ? 'http://localhost:5000/api' 
  : `${window.location.origin}/api`;

/**
 * Initializes the blog page by loading blog posts when the DOM is fully loaded.
 */
document.addEventListener('DOMContentLoaded', function() {
  loadBlogPosts();

  // Add event listener for search input (search on Enter key)
  const searchInput = document.getElementById('searchInput');
  if (searchInput) {
    searchInput.addEventListener('keypress', function(e) {
      if (e.key === 'Enter') {
        searchPosts();
      }
    });
  }

  // Add event listener for category filter change
  const categoryFilter = document.getElementById('categoryFilter');
  if (categoryFilter) {
    categoryFilter.addEventListener('change', filterByCategory);
  }

  // Add event listener for clear filters button
  const clearFiltersBtn = document.getElementById('clearFiltersBtn');
  if (clearFiltersBtn) {
    clearFiltersBtn.addEventListener('click', clearFilters);
  }
});

/**
 * Fetches blog posts from the backend API.
 * Updates `allPosts` and `filteredPosts` arrays upon successful fetch.
 * Calls `displayPosts` and `setupPagination` to render the UI.
 * Handles errors during the API call and displays an appropriate message.
 */
async function loadBlogPosts() {
  try {
    console.log('Loading blog posts from:', `${API_BASE_URL}/blogposts`);
    
    // Show loading spinner while fetching data
    const loadingSpinner = document.getElementById('loadingSpinner');
    if (loadingSpinner) {
      loadingSpinner.style.display = 'flex'; // Use flex to center spinner
    }

    const response = await fetch(`${API_BASE_URL}/blogposts`, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
      }
    });
    
    if (!response.ok) {
      // If response is not OK (e.g., 404, 500), throw an error
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    
    const data = await response.json();
    console.log('Blog posts loaded:', data);
    
    // Ensure data is an array; if not, initialize as empty array
    allPosts = Array.isArray(data) ? data : [];
    filteredPosts = [...allPosts]; // Initialize filtered posts with all posts
    
    displayPosts(); // Render posts to the UI
    setupPagination(); // Set up pagination controls
    
  } catch (error) {
    console.error('Error loading blog posts:', error);
    displayErrorMessage(); // Show error message to the user
  } finally {
    // Hide loading spinner after fetch completes (whether success or error)
    const loadingSpinner = document.getElementById('loadingSpinner');
    if (loadingSpinner) {
      loadingSpinner.style.display = 'none';
    }
  }
}

/**
 * Displays a subset of `filteredPosts` based on the current page.
 * Clears existing posts and dynamically creates new post elements.
 * Initializes AOS (Animate On Scroll) for newly added elements.
 */
function displayPosts() {
  const container = document.getElementById('blogPostsContainer');
  
  // Clear existing posts in the container
  if (container) {
    container.innerHTML = '';
  } else {
    console.error('Blog posts container not found.');
    return;
  }
  
  if (filteredPosts.length === 0) {
    displayNoPostsMessage(); // Show message if no posts are found
    return;
  }
  
  // Calculate the start and end indices for posts on the current page
  const startIndex = (currentPage - 1) * postsPerPage;
  const endIndex = startIndex + postsPerPage;
  const postsToShow = filteredPosts.slice(startIndex, endIndex);
  
  // Create and append HTML for each post
  postsToShow.forEach((post, index) => {
    const postElement = createPostElement(post, index);
    container.appendChild(postElement);
  });
  
  // Refresh AOS to apply animations to newly added elements
  if (typeof AOS !== 'undefined') {
    AOS.refresh();
  }
}

/**
 * Creates an HTML article element for a single blog post.
 * @param {Object} post - The blog post data object.
 * @param {number} index - The index of the post (used for AOS animation delay).
 * @returns {HTMLElement} The created article element.
 */
function createPostElement(post, index) {
  const col = document.createElement('div');
  col.className = 'col-lg-4 col-md-6';
  // Add AOS attributes for staggered animation
  col.setAttribute('data-aos', 'fade-up');
  col.setAttribute('data-aos-delay', (index % 3) * 100); // Stagger delay for visual effect
  
  // Format the post date
  const postDate = new Date(post.date);
  const formattedDate = postDate.toLocaleDateString('en-US', {
    year: 'numeric',
    month: 'long',
    day: 'numeric'
  });
  
  // Generate a short excerpt from the post content
  const excerpt = post.content ? post.content.substring(0, 150) + '...' : 'No content available.';
  
  // Determine category and assign a Bootstrap color class based on tags
  let category = 'General';
  let categoryColor = 'bg-secondary'; // Default color
  
  if (post.tags) {
    const tags = post.tags.toLowerCase();
    if (tags.includes('research')) {
      category = 'Research';
      categoryColor = 'bg-primary';
    } else if (tags.includes('technology')) {
      category = 'Technology';
      categoryColor = 'bg-success';
    } else if (tags.includes('agriculture')) {
      category = 'Agriculture';
      categoryColor = 'bg-warning';
    } else if (tags.includes('prediction')) {
      category = 'Prediction';
      categoryColor = 'bg-info';
    } else if (tags.includes('news')) {
      category = 'News';
      categoryColor = 'bg-danger';
    }
  } else if (post.region) {
    // Fallback to region if no specific tags are found
    category = post.region;
    categoryColor = 'bg-info';
  }
  
  // Handle post image: use provided URL or a default icon
  let imageHtml = '';
  if (post.image_url) {
    imageHtml = `
      <img src="${post.image_url}" class="card-img-top" alt="${post.title}" style="height: 200px; object-fit: cover;">
    `;
  } else {
    imageHtml = `
      <div class="card-img-top bg-gradient-primary d-flex align-items-center justify-content-center text-white" style="height: 200px;">
        <i class="bi bi-journal-text" style="font-size: 3rem;"></i>
      </div>
    `;
  }
  
  // Construct the inner HTML for the blog post card
  col.innerHTML = `
    <article class="card h-100 shadow-sm blog-post-card">
      ${imageHtml}
      <div class="card-body d-flex flex-column">
        <div class="mb-2">
          <span class="badge ${categoryColor}">${category}</span>
          ${post.region ? `<span class="badge bg-outline-secondary ms-1">${post.region}</span>` : ''}
          ${post.country ? `<span class="badge bg-outline-secondary ms-1">${post.country}</span>` : ''}
        </div>
        <h5 class="card-title">
          <a href="#" onclick="openPostModal(${post.id})" class="text-decoration-none text-dark">
            ${post.title}
          </a>
        </h5>
        <p class="card-text text-muted flex-grow-1">${excerpt}</p>
        <div class="card-meta d-flex justify-content-between align-items-center text-muted small mb-3">
          <span><i class="bi bi-person me-1"></i>${post.author || 'LocustHub Team'}</span>
          <span><i class="bi bi-calendar me-1"></i>${formattedDate}</span>
        </div>
        <a href="#" onclick="openPostModal(${post.id})" class="btn btn-primary btn-sm align-self-start">
          Read More <i class="bi bi-arrow-right ms-1"></i>
        </a>
      </div>
    </article>
  `;
  
  return col;
}

/**
 * Displays an error message in the blog posts container if fetching fails.
 */
function displayErrorMessage() {
  const container = document.getElementById('blogPostsContainer');
  if (container) {
    container.innerHTML = `
      <div class="col-12">
        <div class="alert alert-danger text-center" role="alert">
          <i class="bi bi-exclamation-triangle fs-1 d-block mb-3"></i>
          <h4>Unable to Load Blog Posts</h4>
          <p>We're having trouble loading the blog posts right now. Please try again later.</p>
          <button class="btn btn-primary" onclick="loadBlogPosts()">
            <i class="bi bi-arrow-clockwise me-2"></i>Try Again
          </button>
        </div>
      </div>
    `;
  }
}

/**
 * Displays a message indicating no blog posts were found (e.g., after filtering).
 */
function displayNoPostsMessage() {
  const container = document.getElementById('blogPostsContainer');
  if (container) {
    container.innerHTML = `
      <div class="col-12">
        <div class="text-center py-5">
          <i class="bi bi-journal-x text-muted" style="font-size: 5rem;"></i>
          <h3 class="mt-3 text-muted">No Blog Posts Found</h3>
          <p class="text-muted">There are no blog posts available matching your criteria. Check back later for updates or try different filters!</p>
          <button class="btn btn-primary" onclick="clearFilters()">
            <i class="bi bi-arrow-clockwise me-2"></i>Refresh
          </button>
        </div>
      </div>
    `;
  }
}

/**
 * Sets up the pagination controls based on the number of filtered posts.
 * Dynamically creates page links and handles active state.
 */
function setupPagination() {
  const totalPages = Math.ceil(filteredPosts.length / postsPerPage);
  const paginationContainer = document.getElementById('pagination');
  
  if (!paginationContainer) {
    console.error('Pagination container not found.');
    return;
  }

  // Hide pagination if there's only one page or no posts
  if (totalPages <= 1) {
    paginationContainer.innerHTML = '';
    return;
  }
  
  let paginationHTML = '<ul class="pagination">';
  
  // Previous button
  paginationHTML += `
    <li class="page-item ${currentPage === 1 ? 'disabled' : ''}">
      <a class="page-link" href="#" onclick="changePage(${currentPage - 1})" aria-label="Previous">
        <i class="bi bi-chevron-left"></i>
      </a>
    </li>
  `;
  
  // Page numbers (displaying a limited range around the current page)
  const maxPagesToShow = 5; // Max number of page buttons to show
  let startPage = Math.max(1, currentPage - Math.floor(maxPagesToShow / 2));
  let endPage = Math.min(totalPages, startPage + maxPagesToShow - 1);

  if (endPage - startPage + 1 < maxPagesToShow) {
    startPage = Math.max(1, endPage - maxPagesToShow + 1);
  }

  if (startPage > 1) {
    paginationHTML += `
      <li class="page-item">
        <a class="page-link" href="#" onclick="changePage(1)">1</a>
      </li>
    `;
    if (startPage > 2) {
      paginationHTML += `<li class="page-item disabled"><span class="page-link">...</span></li>`;
    }
  }

  for (let i = startPage; i <= endPage; i++) {
    paginationHTML += `
      <li class="page-item ${i === currentPage ? 'active' : ''}">
        <a class="page-link" href="#" onclick="changePage(${i})">${i}</a>
      </li>
    `;
  }

  if (endPage < totalPages) {
    if (endPage < totalPages - 1) {
      paginationHTML += `<li class="page-item disabled"><span class="page-link">...</span></li>`;
    }
    paginationHTML += `
      <li class="page-item">
        <a class="page-link" href="#" onclick="changePage(${totalPages})">${totalPages}</a>
      </li>
    `;
  }
  
  // Next button
  paginationHTML += `
    <li class="page-item ${currentPage === totalPages ? 'disabled' : ''}">
      <a class="page-link" href="#" onclick="changePage(${currentPage + 1})" aria-label="Next">
        <i class="bi bi-chevron-right"></i>
      </a>
    </li>
  `;
  
  paginationHTML += '</ul>';
  paginationContainer.innerHTML = paginationHTML;
}

/**
 * Changes the current page and re-renders the blog posts and pagination.
 * Scrolls the user to the top of the blog section for better UX.
 * @param {number} page - The page number to navigate to.
 */
function changePage(page) {
  if (page < 1 || page > Math.ceil(filteredPosts.length / postsPerPage)) {
    return; // Prevent navigating to invalid pages
  }
  currentPage = page;
  displayPosts();
  setupPagination();
  
  // Scroll to top of blog section for better user experience
  document.getElementById('blog').scrollIntoView({ behavior: 'smooth' });
}

/**
 * Filters blog posts based on the search input value.
 * Resets the current page to 1 and updates the displayed posts and pagination.
 */
function searchPosts() {
  const searchTerm = document.getElementById('searchInput').value.toLowerCase().trim();
  
  if (searchTerm === '') {
    filteredPosts = [...allPosts]; // If search term is empty, show all posts
  } else {
    // Filter posts if title, content, tags, author, region, or country contain the search term
    filteredPosts = allPosts.filter(post => 
      post.title.toLowerCase().includes(searchTerm) ||
      (post.content && post.content.toLowerCase().includes(searchTerm)) ||
      (post.tags && post.tags.toLowerCase().includes(searchTerm)) ||
      (post.author && post.author.toLowerCase().includes(searchTerm)) ||
      (post.region && post.region.toLowerCase().includes(searchTerm)) ||
      (post.country && post.country.toLowerCase().includes(searchTerm))
    );
  }
  
  currentPage = 1; // Reset to the first page after search
  displayPosts();
  setupPagination();
}

/**
 * Filters blog posts based on the selected category from the dropdown.
 * Resets the current page to 1 and updates the displayed posts and pagination.
 */
function filterByCategory() {
  const selectedCategory = document.getElementById('categoryFilter').value;
  
  if (selectedCategory === 'all') {
    filteredPosts = [...allPosts]; // Show all posts if 'All Categories' is selected
  } else {
    // Filter posts by checking if their tags include the selected category
    filteredPosts = allPosts.filter(post => {
      if (post.tags) {
        return post.tags.toLowerCase().includes(selectedCategory.toLowerCase());
      }
      return false; // Exclude posts without tags
    });
  }
  
  currentPage = 1; // Reset to the first page after filtering
  displayPosts();
  setupPagination();
}

/**
 * Clears all active search and filter criteria.
 * Resets the UI to display all original blog posts.
 */
function clearFilters() {
  document.getElementById('searchInput').value = ''; // Clear search input
  document.getElementById('categoryFilter').value = 'all'; // Reset category filter
  filteredPosts = [...allPosts]; // Restore all original posts
  currentPage = 1; // Reset to the first page
  displayPosts();
  setupPagination();
}

/**
 * Expose functions to the global scope if needed for inline event handlers
 * (e.g., onclick="openPostModal(id)")
 */
window.loadBlogPosts = loadBlogPosts;
window.searchPosts = searchPosts;
window.filterByCategory = filterByCategory;
window.clearFilters = clearFilters;
window.changePage = changePage;
window.openPostModal = openPostModal;

/**
 * Opens a modal displaying the full content of a selected blog post.
 * Fetches the post details if not already loaded and displays them in a modal.
 * @param {number|string} postId - The ID of the blog post to display.
 */
async function openPostModal(postId) {
  try {
    // Show loading state in modal
    const modal = document.getElementById('blogPostModal');
    const modalContent = document.getElementById('blogPostModalBody');
    
    if (!modal || !modalContent) {
      console.error('Required modal elements not found');
      return;
    }
    
    // Set loading state
    modalContent.innerHTML = `
      <div class="text-center py-5">
        <div class="spinner-border text-primary" role="status">
          <span class="visually-hidden">Loading...</span>
        </div>
        <p class="mt-2">Loading post...</p>
      </div>
    `;
    
    // Show the modal immediately with loading state
    const modalInstance = new bootstrap.Modal(modal);
    modalInstance.show();
    
    // Try to find post in existing data first
    let post = allPosts.find(p => p.id == postId);
    
    // If post not found, fetch it from the API
    if (!post) {
      const response = await fetch(`${API_BASE_URL}/blogposts/${postId}`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json'
        }
      });
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      post = await response.json();
    }
    
    if (!post) {
      throw new Error('Post not found');
    }
    
    // Format the date
    const postDate = post.date ? new Date(post.date) : new Date();
    const options = { year: 'numeric', month: 'long', day: 'numeric' };
    const formattedDate = postDate.toLocaleDateString('en-US', options);
    
    // Determine reading time (approximately 200 words per minute)
    const wordCount = post.content ? post.content.split(/\s+/).length : 0;
    const readingTime = Math.ceil(wordCount / 200);
    
    // Generate a brief excerpt if not provided
    const excerpt = post.excerpt || (post.content 
      ? post.content.substring(0, 200).replace(/<[^>]*>?/gm, '') + '...' 
      : 'No excerpt available.');
    
    // Generate tags if not provided
    const tags = post.tags ? post.tags.split(',').map(tag => tag.trim()) : [];
    
    // Create the modal content
    modalContent.innerHTML = `
      <div class="container-fluid">
        <div class="row justify-content-center">
          <!-- Main Content -->
          <div class="col-lg-8">
            <article class="blog-post">
              <!-- Featured Image -->
              ${post.image_url ? `
                <div class="blog-post-featured-image mb-4">
                  <img src="${post.image_url}" alt="${post.title || ''}" class="img-fluid rounded-3 w-100" style="max-height: 450px; object-fit: cover;">
                </div>
              ` : ''}
              
              <!-- Post Header -->
              <header class="blog-post-header mb-4">
                <div class="d-flex align-items-center mb-3">
                  <span class="badge bg-primary me-2">${post.region || 'Uncategorized'}</span>
                  <span class="text-muted">
                    <i class="far fa-clock me-1"></i> ${readingTime} min read
                  </span>
                </div>
                <h1 class="blog-post-title display-5 fw-bold mb-3">${post.title || 'Untitled Post'}</h1>
                <div class="d-flex align-items-center text-muted mb-4">
                  <div class="d-flex align-items-center me-4">
                    <i class="far fa-user me-2"></i>
                    <span>${post.author || 'LocustHub Team'}</span>
                  </div>
                  <div class="d-flex align-items-center">
                    <i class="far fa-calendar-alt me-2"></i>
                    <span>${formattedDate}</span>
                  </div>
                </div>
              </header>
              
              <!-- Post Content -->
              <div class="blog-post-content">
                <div class="lead mb-4">${excerpt}</div>
                <hr class="my-4">
                <div class="blog-post-body">
                  ${post.content || '<p class="text-muted">No content available for this post.</p>'}
                </div>
                
                <!-- Tags -->
                ${tags.length > 0 ? `
                  <div class="post-tags mt-5 pt-4 border-top">
                    <h6 class="d-inline-block me-2 mb-0">Tags:</h6>
                    <div class="d-inline-flex flex-wrap gap-2">
                      ${tags.map(tag => `
                        <a href="#" class="badge bg-light text-dark text-decoration-none" onclick="filterByTag('${tag}'); return false;">
                          #${tag}
                        </a>
                      `).join('')}
                    </div>
                  </div>
                ` : ''}
                
                <!-- Share Buttons -->
                <div class="share-buttons mt-4 pt-4 border-top">
                  <h6 class="mb-3">Share this post:</h6>
                  <div class="d-flex gap-2">
                    <a href="#" class="btn btn-outline-primary btn-sm rounded-circle share-facebook" data-bs-toggle="tooltip" title="Share on Facebook">
                      <i class="fab fa-facebook-f"></i>
                    </a>
                    <a href="#" class="btn btn-outline-info btn-sm rounded-circle share-twitter" data-bs-toggle="tooltip" title="Share on Twitter">
                      <i class="fab fa-twitter"></i>
                    </a>
                    <a href="#" class="btn btn-outline-danger btn-sm rounded-circle share-linkedin" data-bs-toggle="tooltip" title="Share on LinkedIn">
                      <i class="fab fa-linkedin-in"></i>
                    </a>
                    <a href="#" class="btn btn-outline-secondary btn-sm rounded-circle share-copy" data-bs-toggle="tooltip" title="Copy link">
                      <i class="fas fa-link"></i>
                    </a>
                  </div>
                </div>
              </div>
            </article>
            
            <!-- Author Bio -->
            <div class="author-bio card mt-5">
              <div class="card-body">
                <div class="d-flex align-items-center">
                  <img src="assets/img/team/default-avatar.jpg" alt="Author" class="rounded-circle me-3" width="80" height="80">
                  <div>
                    <h5 class="card-title mb-1">${post.author || 'LocustHub Team'}</h5>
                    <p class="text-muted small mb-2">Agricultural Scientist & Researcher</p>
                    <p class="card-text small">${post.author_bio || 'Expert in locust research and agricultural technology.'}</p>
                    <div class="author-social-links">
                      <a href="#" class="text-muted me-2"><i class="fab fa-twitter"></i></a>
                      <a href="#" class="text-muted me-2"><i class="fab fa-linkedin-in"></i></a>
                      <a href="#" class="text-muted"><i class="fas fa-globe"></i></a>
                    </div>
                  </div>
                </div>
              </div>
            </div>
            
            <!-- Comments Section -->
            <div class="comments-section mt-5 pt-4 border-top">
              <h5 class="mb-4">Comments</h5>
              <div id="disqus_thread"></div>
            </div>
          </div>
          
          <!-- Sidebar -->
          <div class="col-lg-4">
            <!-- Related Posts -->
            <div class="card mb-4">
              <div class="card-header bg-light">
                <h6 class="mb-0">Related Posts</h6>
              </div>
              <div class="card-body p-0">
                <div class="list-group list-group-flush" id="related-posts">
                  ${generateRelatedPosts(post.id, post.region).slice(0, 3).map(relatedPost => `
                    <a href="#" class="list-group-item list-group-item-action border-0 py-3" onclick="openPostModal(${relatedPost.id}); return false;">
                      <div class="d-flex align-items-center">
                        ${relatedPost.image_url ? `
                          <img src="${relatedPost.image_url}" alt="${relatedPost.title}" class="rounded me-3" width="80" height="60" style="object-fit: cover;">
                        ` : ''}
                        <div>
                          <h6 class="mb-1">${relatedPost.title}</h6>
                          <small class="text-muted">${new Date(relatedPost.date).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })}</small>
                        </div>
                      </div>
                    </a>
                  `).join('')}
                </div>
              </div>
            </div>
            
            <!-- Newsletter Signup -->
            <div class="card mb-4">
              <div class="card-body">
                <h6 class="card-title">Subscribe to our newsletter</h6>
                <p class="small text-muted">Get the latest posts delivered right to your inbox.</p>
                <form class="mt-3">
                  <div class="input-group mb-2">
                    <input type="email" class="form-control form-control-sm" placeholder="Your email address" required>
                    <button class="btn btn-primary btn-sm" type="submit">Subscribe</button>
                  </div>
                  <div class="form-check small">
                    <input class="form-check-input" type="checkbox" id="privacyPolicy" required>
                    <label class="form-check-label" for="privacyPolicy">
                      I agree to the privacy policy
                    </label>
                  </div>
                </form>
              </div>
            </div>
          </div>
        </div>
      </div>
    `;
    
    // Initialize any plugins or components that need to run after content is loaded
    initializeBlogContent(modalContent);
    
    // Load Disqus comments
    loadDisqusComments(post.id, post.title);
    
  } catch (error) {
    console.error('Error loading blog post:', error);
    const modalContent = document.getElementById('blogPostModalBody');
    if (modalContent) {
      modalContent.innerHTML = `
        <div class="alert alert-danger" role="alert">
          <h4 class="alert-heading">Error Loading Post</h4>
          <p>We couldn't load the requested blog post. Please try again later.</p>
          <hr>
          <p class="mb-0">${error.message || 'Unknown error occurred'}</p>
        </div>
      `;
    }
  }
}

/**
 * Generate related posts based on the current post's category/tags
 * @param {string} currentPostId - ID of the current post to exclude
 * @param {string} category - Category to find related posts in
 * @returns {Array} Array of related posts
 */
function generateRelatedPosts(currentPostId, category) {
  if (!allPosts || allPosts.length === 0) return [];
  
  // Filter out the current post and get posts from the same category
  return allPosts
    .filter(post => post.id !== currentPostId && post.region === category)
    .sort(() => 0.5 - Math.random()) // Shuffle array
    .slice(0, 5); // Limit to 5 related posts
}

/**
 * Load Disqus comments
 * @param {string} postId - Post ID for Disqus thread
 * @param {string} postTitle - Post title for Disqus thread
 */
function loadDisqusComments(postId, postTitle) {
  if (window.DISQUS) {
    // If Disqus is already loaded, reset it
    DISQUS.reset({
      reload: true,
      config: function() {
        this.page.identifier = `blog-${postId}`;
        this.page.url = window.location.href.split('?')[0] + `?post=${postId}`;
        this.page.title = `${postTitle} | LocustHub Blog`;
      }
    });
  } else {
    // Otherwise, configure and load Disqus
    window.disqus_config = function() {
      this.page.identifier = `blog-${postId}`;
      this.page.url = window.location.href.split('?')[0] + `?post=${postId}`;
      this.page.title = `${postTitle} | LocustHub Blog`;
    };
    
    const d = document, s = d.createElement('script');
    s.src = 'https://locusthub.disqus.com/embed.js';
    s.setAttribute('data-timestamp', +new Date());
    (d.head || d.body).appendChild(s);
  }
}

/**
 * Filter posts by tag
 * @param {string} tag - Tag to filter by
 */
function filterByTag(tag) {
  const searchInput = document.getElementById('searchInput');
  if (searchInput) {
    searchInput.value = tag;
    searchPosts();
  }
}

/**
 * Initialize any plugins or components within the blog content
 * @param {HTMLElement} container - The container element to initialize components in
 */
function initializeBlogContent(container) {
  if (!container) return;
  
  // Initialize tooltips
  const tooltipTriggerList = [].slice.call(container.querySelectorAll('[data-bs-toggle="tooltip"]'));
  tooltipTriggerList.map(function (tooltipTriggerEl) {
    return new bootstrap.Tooltip(tooltipTriggerEl);
  });
  
  // Make images responsive
  container.querySelectorAll('img:not(.img-fluid)').forEach(img => {
    if (!img.classList.contains('img-fluid')) {
      img.classList.add('img-fluid');
    }
  });
  
  // Add table responsive wrapper
  container.querySelectorAll('table').forEach(table => {
    if (!table.closest('.table-responsive')) {
      const wrapper = document.createElement('div');
      wrapper.className = 'table-responsive';
      table.parentNode.insertBefore(wrapper, table);
      wrapper.appendChild(table);
    }
  });
}
