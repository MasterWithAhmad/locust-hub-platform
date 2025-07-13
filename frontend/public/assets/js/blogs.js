// Initialize Disqus when document is ready
document.addEventListener("DOMContentLoaded", function () {
  if (document.getElementById("disqus_thread")) {
    // Disqus configuration
    var disqus_config = function () {
      this.page.url = window.location.href;
      this.page.identifier = window.location.pathname;
    };

    // Load Disqus script
    (function () {
      var d = document,
        s = d.createElement("script");
      s.src = "https://locusthub.disqus.com/embed.js";
      s.setAttribute("data-timestamp", +new Date());
      (d.head || d.body).appendChild(s);
    })();
  }
});

// Global variables
let allPosts = [];
let filteredPosts = [];
let currentPage = 1;
const postsPerPage = 6;


// Initialize the blog page
document.addEventListener("DOMContentLoaded", function () {
  loadBlogPosts();
});

// Load blog posts from the API
async function loadBlogPosts() {
  try {
    console.log("1. Starting to load blog posts...");
    
    // Check if API is available
    if (!window.api || !window.api.blog) {
      console.error("API not available!");
      throw new Error("API not initialized");
    }
    
    console.log("2. API is available, fetching public posts...");
    
    // Use the public endpoint to get all blog posts
    try {
      console.log("3. Calling getPublicPosts()...");
      const posts = await window.api.blog.getPublicPosts();
      console.log("4. Received response:", posts);
      
      if (posts && Array.isArray(posts)) {
        console.log(`5. Successfully loaded ${posts.length} posts`);
        processBlogPosts(posts);
        return;
      } else {
        console.error("6. Invalid posts data received:", posts);
        throw new Error("Invalid posts data format");
      }
    } catch (error) {
      console.error("7. Error in getPublicPosts():", error);
      throw error;
    }
  } catch (error) {
    console.error("8. Fatal error in loadBlogPosts():", error);
    displayErrorMessage();
  }
}

// Process and display blog posts
function processBlogPosts(posts) {
  console.log("Blog posts loaded:", posts);
  
  // Debug: print the actual structure
  if (Array.isArray(posts) && posts.length > 0) {
    console.log("First post structure:", JSON.stringify(posts[0], null, 2));
  }

  allPosts = Array.isArray(posts) ? posts : [];
  filteredPosts = [...allPosts];

  displayPosts();
  setupPagination();
}

// Display blog posts
function displayPosts() {
  const container = document.getElementById("blogPostsContainer");
  const loadingSpinner = document.getElementById("loadingSpinner");

  // Hide loading spinner if it exists
  if (loadingSpinner) {
    loadingSpinner.style.display = "none";
  }

  // Check if container exists
  if (!container) {
    console.error("Blog posts container not found");
    document.body.innerHTML += '<div style="color:red;font-size:2em;">blogPostsContainer NOT FOUND</div>';
    return;
  }

  // Clear container safely
  container.innerHTML = "";

  if (!Array.isArray(filteredPosts) || filteredPosts.length === 0) {
    displayNoPostsMessage();
    return;
  }

  // Calculate pagination
  const startIndex = (currentPage - 1) * postsPerPage;
  const endIndex = startIndex + postsPerPage;
  const postsToShow = filteredPosts.slice(startIndex, endIndex);

  // Generate HTML for each post
  postsToShow.forEach((post, index) => {
    const postElement = createPostElement(post, index);
    console.log("Appending card for post:", post.title || post.id);
    container.appendChild(postElement);
    console.log("Appended card for post:", post.title || post.id);
  });

  // Initialize AOS for new elements
  if (typeof AOS !== "undefined") {
    AOS.refresh();
  }
}

// Create a single post element
function createPostElement(post, index) {
  const col = document.createElement('div');
  col.className = 'col-lg-4 col-md-6';
  col.setAttribute('data-aos', 'fade-up');
  col.setAttribute('data-aos-delay', (index % 3) * 100);

  const excerpt = post.content ? post.content.replace(/<[^>]+>/g, '').slice(0, 120) + (post.content.replace(/<[^>]+>/g, '').length > 120 ? '...' : '') : 'No content available.';
  const avatar = getAuthorAvatar(post.author);
  const formattedDate = formatDate(post.date);
  const tag = post.country || post.region || 'General';
  // Use a local fallback image if the main image fails to load
  const fallbackImage = '/assets/blog_images/blog_c958fcef91374d64ad27ea17feebdce2.webp';
  const image = post.image_url || post.imageUrl || fallbackImage;

  // Prepare tags as badges
  let tagsHtml = '';
  if (post.tags) {
    const tagsArr = post.tags.split(',').map(t => t.trim()).filter(Boolean);
    tagsHtml = tagsArr.map(t => `<span class="badge bg-light text-dark border border-primary ms-1">#${t}</span>`).join(' ');
  }

  // Unique modal id for each post
  const modalId = `blogModal_${post.id}`;

  col.innerHTML = `
    <div class="blog-post-card card mb-4 fade-in h-100">
      <div class="card-img-container position-relative" style="height: 200px; overflow: hidden;">
        <img src="${image}" class="card-img-top h-100 w-100" alt="${post.title}" style="object-fit: cover;" onerror="this.onerror=null; this.src='${fallbackImage}'">
        <div class="img-gradient-overlay"></div>
        <span class="badge bg-primary position-absolute top-0 start-0 m-3">${tag}</span>
        <div class="position-absolute bottom-0 start-0 w-100 p-3 text-white" style="background: linear-gradient(transparent, rgba(0,0,0,0.7));">
          <h5 class="card-title mb-1 text-white">${post.title}</h5>
          <div class="d-flex align-items-center small">
            ${avatar}
            <span class="ms-2">${post.author || 'Unknown'}</span>
          </div>
        </div>
      </div>
      <div class="card-body d-flex flex-column">
        <div class="blog-post-excerpt mb-3">${excerpt}</div>
        <div class="d-flex justify-content-between align-items-center mt-auto">
          <div class="blog-post-meta d-flex align-items-center">
            <span class="text-muted small"><i class="bi bi-calendar me-1"></i>${formattedDate}</span>
          </div>
          <button class="btn btn-sm btn-outline-primary view-btn" data-bs-toggle="modal" data-bs-target="#${modalId}">
            Read More <i class="bi bi-arrow-right ms-1"></i>
          </button>
        </div>
      </div>
    </div>

    <!-- Enhanced Modal -->
    <div class="modal fade" id="${modalId}" tabindex="-1" aria-labelledby="${modalId}_label" aria-hidden="true">
      <div class="modal-dialog modal-xl modal-dialog-centered modal-dialog-scrollable">
        <div class="modal-content border-0 shadow-lg">
          <!-- Modal Header -->
          <div class="modal-header border-0 pb-0 position-sticky top-0 bg-white" style="z-index: 1050;">
            <div>
              <div class="d-flex align-items-center mb-2">
                ${avatar}
                <div class="ms-2">
                  <div class="fw-medium">${post.author || 'Unknown'}</div>
                  <div class="text-muted small">${formattedDate}</div>
                </div>
              </div>
              <h2 class="modal-title h3 mb-0" id="${modalId}_label">${post.title}</h2>
            </div>
            <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
          </div>

          <!-- Modal Body -->
          <div class="modal-body py-4">
            ${post.image_url ? `
              <div class="text-center mb-4">
                <img src="${post.image_url}" class="img-fluid rounded-3 shadow" alt="${post.title}" onerror="this.onerror=null; this.src='${fallbackImage}'">
              </div>
            ` : ''}
            
            <article class="blog-content" style="line-height: 1.8; font-size: 1.1rem;">
              ${post.content || '<p class="text-muted">No content available.</p>'}
            </article>

            ${tagsHtml ? `
              <div class="mt-5 pt-3 border-top">
                <h6 class="mb-3">Tags</h6>
                <div class="d-flex flex-wrap gap-2">
                  ${tagsHtml}
                </div>
              </div>
            ` : ''}
          </div>

          <!-- Modal Footer -->
          <div class="modal-footer border-0 bg-light">
            <div class="d-flex justify-content-between w-100 align-items-center">
              <div class="social-share">
                <span class="text-muted me-2">Share:</span>
                <a href="#" class="text-muted me-3" onclick="shareOnTwitter('${post.title}', window.location.href); return false;">
                  <i class="bi bi-twitter-x"></i>
                </a>
                <a href="#" class="text-muted me-3" onclick="shareOnFacebook(window.location.href); return false;">
                  <i class="bi bi-facebook"></i>
                </a>
                <a href="#" class="text-muted" onclick="shareOnLinkedIn('${post.title}', window.location.href); return false;">
                  <i class="bi bi-linkedin"></i>
                </a>
              </div>
              <div>
                <button class="btn btn-sm btn-outline-secondary me-2" title="Save for later">
                  <i class="bi bi-bookmark"></i> Save
                </button>
                <button class="btn btn-primary" data-bs-dismiss="modal">Close</button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  `;

  // Add social sharing functions to window object if they don't exist
  if (!window.shareOnTwitter) {
    window.shareOnTwitter = function(title, url) {
      window.open(`https://twitter.com/intent/tweet?text=${encodeURIComponent(title)}&url=${encodeURIComponent(url)}`, '_blank');
      return false;
    };
  }

  if (!window.shareOnFacebook) {
    window.shareOnFacebook = function(url) {
      window.open(`https://www.facebook.com/sharer/sharer.php?u=${encodeURIComponent(url)}`, '_blank');
      return false;
    };
  }

  if (!window.shareOnLinkedIn) {
    window.shareOnLinkedIn = function(title, url) {
      window.open(`https://www.linkedin.com/shareArticle?mini=true&url=${encodeURIComponent(url)}&title=${encodeURIComponent(title)}`, '_blank');
      return false;
    };
  }
  return col;
}

// Display error message
function displayErrorMessage() {
  const container = document.getElementById("blogPostsContainer");
  const errorMessage = `
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

  if (container) {
    container.innerHTML = errorMessage;
  } else {
    // If container doesn't exist, create it and append to body
    const newContainer = document.createElement("div");
    newContainer.id = "blogPostsContainer";
    newContainer.className = "container mt-4";
    newContainer.innerHTML = errorMessage;
    const mainContent = document.querySelector("main") || document.body;
    mainContent.appendChild(newContainer);
  }
}

// Display no posts message
function displayNoPostsMessage() {
  const container = document.getElementById("blogPostsContainer");
  const noPostsMessage = `
        <div class="col-12">
          <div class="text-center py-5">
            <i class="bi bi-journal-x text-muted" style="font-size: 5rem;"></i>
            <h3 class="mt-3 text-muted">No Blog Posts Found</h3>
            <p class="text-muted">There are no blog posts available matching your criteria.</p>
            <button class="btn btn-primary" onclick="clearFilters()">
              <i class="bi bi-arrow-clockwise me-2"></i>Clear Filters
            </button>
          </div>
        </div>
      `;

  if (container) {
    container.innerHTML = noPostsMessage;
  } else {
    console.warn(
      "Blog posts container not found for displaying no posts message"
    );
  }
}

// Setup pagination
function setupPagination() {
  if (!Array.isArray(filteredPosts)) {
    console.warn("filteredPosts is not an array");
    return;
  }

  const totalPages = Math.ceil(filteredPosts.length / postsPerPage);
  const paginationContainer = document.getElementById("pagination");

  if (!paginationContainer) {
    console.warn("Pagination container not found");
    return;
  }

  if (totalPages <= 1) {
    paginationContainer.innerHTML = "";
    return;
  }

  let paginationHTML = "";

  // Previous button
  if (currentPage > 1) {
    paginationHTML += `
          <li class="page-item">
            <a class="page-link" href="#" onclick="changePage(${
              currentPage - 1
            }); return false;">
              <i class="bi bi-chevron-left"></i>
            </a>
          </li>
        `;
  }

  // Page numbers
  for (let i = 1; i <= totalPages; i++) {
    if (i === currentPage) {
      paginationHTML += `
            <li class="page-item active">
              <span class="page-link">${i}</span>
            </li>
          `;
    } else if (
      i === 1 ||
      i === totalPages ||
      (i >= currentPage - 2 && i <= currentPage + 2)
    ) {
      paginationHTML += `
            <li class="page-item">
              <a class="page-link" href="#" onclick="changePage(${i})">${i}</a>
            </li>
          `;
    } else if (i === currentPage - 3 || i === currentPage + 3) {
      paginationHTML += `
            <li class="page-item disabled">
              <span class="page-link">...</span>
            </li>
          `;
    }
  }

  // Next button
  if (currentPage < totalPages) {
    paginationHTML += `
          <li class="page-item">
            <a class="page-link" href="#" onclick="changePage(${
              currentPage + 1
            })">
              <i class="bi bi-chevron-right"></i>
            </a>
          </li>
        `;
  }

  paginationContainer.innerHTML = paginationHTML;
}

// Change page
function changePage(page) {
  currentPage = page;
  displayPosts();
  setupPagination();

  // Scroll to top of blog section
  document.getElementById("blog").scrollIntoView({ behavior: "smooth" });
}

// Search posts
function searchPosts() {
  const searchTerm = document
    .getElementById("searchInput")
    .value.toLowerCase()
    .trim();

  if (searchTerm === "") {
    filteredPosts = [...allPosts];
  } else {
    filteredPosts = allPosts.filter(
      (post) =>
        post.title.toLowerCase().includes(searchTerm) ||
        (post.content && post.content.toLowerCase().includes(searchTerm)) ||
        (post.tags && post.tags.toLowerCase().includes(searchTerm)) ||
        (post.author && post.author.toLowerCase().includes(searchTerm)) ||
        (post.region && post.region.toLowerCase().includes(searchTerm)) ||
        (post.country && post.country.toLowerCase().includes(searchTerm))
    );
  }

  currentPage = 1;
  displayPosts();
  setupPagination();
}

// Filter by category
function filterByCategory() {
  const selectedCategory = document.getElementById("categoryFilter").value;

  if (selectedCategory === "all") {
    filteredPosts = [...allPosts];
  } else {
    filteredPosts = allPosts.filter((post) => {
      if (post.tags) {
        return post.tags.toLowerCase().includes(selectedCategory.toLowerCase());
      }
      return false;
    });
  }

  currentPage = 1;
  displayPosts();
  setupPagination();
}

// Clear all filters
function clearFilters() {
  document.getElementById("searchInput").value = "";
  document.getElementById("categoryFilter").value = "all";
  filteredPosts = [...allPosts];
  currentPage = 1;
  displayPosts();
  setupPagination();
}

// Open post modal
function openPostModal(postId) {
  const post = allPosts.find((p) => p.id === postId);

  if (!post) {
    console.error("Post not found");
    return;
  }

  // Update modal content
  document.getElementById("blogPostModalLabel").textContent = post.title;

  const modalBody = document.getElementById("blogPostModalBody");

  // Determine category
  let category = "General";
  if (post.tags) {
    const tags = post.tags.toLowerCase();
    if (tags.includes("research")) category = "Research";
    else if (tags.includes("technology")) category = "Technology";
    else if (tags.includes("agriculture")) category = "Agriculture";
    else if (tags.includes("prediction")) category = "Prediction";
    else if (tags.includes("news")) category = "News";
  }

  modalBody.innerHTML = `
        <div class="mb-4">
          <span class="badge bg-primary me-2">${category}</span>
          ${
            post.region
              ? `<span class="badge bg-info me-2">${post.region}</span>`
              : ""
          }
          ${
            post.country
              ? `<span class="badge bg-secondary me-2">${post.country}</span>`
              : ""
          }
          <br><br>
          <small class="text-muted">
            <i class="bi bi-person me-1"></i>${
              post.author || "LocustHub Team"
            } • 
            <i class="bi bi-calendar me-1"></i>${new Date(
              post.date
            ).toLocaleDateString()}
          </small>
        </div>
        ${
          post.image_url
            ? `<img src="${post.image_url}" class="img-fluid mb-4 rounded" alt="${post.title}">`
            : ""
        }
        <div class="post-content" style="line-height: 1.8;">
          ${
            post.content
              ? post.content.replace(/\n/g, "<br>")
              : "Content not available."
          }
        </div>
      `;

  const modalMeta = document.getElementById("modalPostMeta");
  modalMeta.innerHTML = `
        Published on ${new Date(post.date).toLocaleDateString("en-US", {
          year: "numeric",
          month: "long",
          day: "numeric",
        })}
      `;

  // Show modal
  const modal = new bootstrap.Modal(document.getElementById("blogPostModal"));
  modal.show();
}

// Add event listener for search input (search on Enter key)
document.addEventListener("DOMContentLoaded", function () {
  const searchInput = document.getElementById("searchInput");
  if (searchInput) {
    searchInput.addEventListener("keypress", function (e) {
      if (e.key === "Enter") {
        searchPosts();
      }
    });
  }
});

// --- Search Functionality ---
document.addEventListener('DOMContentLoaded', function () {
  const searchInput = document.getElementById('searchInput');
  if (searchInput) {
    searchInput.addEventListener('input', function() {
      const searchTerm = this.value.toLowerCase().trim();
      filteredPosts = allPosts.filter(post =>
        post.title.toLowerCase().includes(searchTerm) ||
        (post.content && post.content.toLowerCase().includes(searchTerm)) ||
        (post.tags && post.tags.toLowerCase().includes(searchTerm)) ||
        (post.author && post.author.toLowerCase().includes(searchTerm)) ||
        (post.region && post.region.toLowerCase().includes(searchTerm)) ||
        (post.country && post.country.toLowerCase().includes(searchTerm))
      );
      currentPage = 1;
      displayPosts();
      setupPagination();
    });
  }

  // --- Sorting Functionality ---
  const sortSelect = document.getElementById('sortSelect');
  if (sortSelect) {
    sortSelect.addEventListener('change', function() {
      const sortValue = this.value;
      if (sortValue === 'latest') {
        filteredPosts.sort((a, b) => new Date(b.date) - new Date(a.date));
      } else if (sortValue === 'oldest') {
        filteredPosts.sort((a, b) => new Date(a.date) - new Date(b.date));
      } else if (sortValue === 'az') {
        filteredPosts.sort((a, b) => a.title.localeCompare(b.title));
      } else if (sortValue === 'za') {
        filteredPosts.sort((a, b) => b.title.localeCompare(a.title));
      }
      currentPage = 1;
      displayPosts();
      setupPagination();
    });
  }
});

function getAuthorAvatar(author) {
  if (!author) return `<div class="avatar-circle"><span>A</span></div>`;
  const initial = author.trim()[0].toUpperCase();
  return `<div class="avatar-circle"><span>${initial}</span></div>`;
}

function formatDate(dateStr) {
  const date = new Date(dateStr);
  return date.toLocaleDateString('en-US', { year: 'numeric', month: 'long', day: 'numeric' });
}

function renderBlogCard(post) {
  const {
    title, content, image_url, author, date, country, region
  } = post;
  const excerpt = content ? (content.replace(/<[^>]+>/g, '').slice(0, 120) + (content.replace(/<[^>]+>/g, '').length > 120 ? '...' : '')) : 'No content';
  const avatar = getAuthorAvatar(author);
  const formattedDate = date ? formatDate(date) : 'No date';
  const tag = country || region || 'General';
  
  // Handle image URL - if it's a relative path, ensure it's correct
  let image = '/assets/blog_images/blog_c958fcef91374d64ad27ea17feebdce2.webp'; // Default image
  
  if (image_url) {
    if (image_url.startsWith('http') || image_url.startsWith('/')) {
      // Use the URL as is if it's absolute or already a root-relative path
      image = image_url;
    } else {
      // Prepend the API path if it's just a filename
      image = `/api/uploads/blog_images/${image_url}`;
    }
  }
  return `
    <div class="blog-post-card card mb-4 fade-in">
      <div class="card-img-container position-relative">
        <img src="${image}" class="card-img-top" alt="${title}">
        <div class="img-gradient-overlay"></div>
        <span class="badge position-absolute top-0 start-0 m-3 blog-post-category">${tag}</span>
      </div>
      <div class="card-body d-flex flex-column">
        <h5 class="card-title blog-post-title">${title}</h5>
        <div class="blog-post-excerpt">${excerpt}</div>
        <div class="blog-post-meta d-flex align-items-center mt-2">
          ${avatar}
          <span class="ms-2">${author || 'Unknown'}</span>
          <span class="mx-2">·</span>
          <span class="blog-post-date"><i class="bi bi-calendar"></i> ${formattedDate}</span>
        </div>
        <div class="blog-post-actions d-flex align-items-center mt-3">
          <button class="btn btn-link p-0 me-3" title="Like"><i class="bi bi-heart"></i></button>
          <button class="btn btn-link p-0 me-3" title="Comment"><i class="bi bi-chat"></i></button>
          <button class="btn btn-link p-0 me-3" title="Bookmark"><i class="bi bi-bookmark"></i></button>
          <div class="ms-auto social-share">
            <a href="#" class="btn btn-link p-0 me-2" title="Share on Twitter"><i class="bi bi-twitter"></i></a>
            <a href="#" class="btn btn-link p-0 me-2" title="Share on Facebook"><i class="bi bi-facebook"></i></a>
            <a href="#" class="btn btn-link p-0" title="Share on LinkedIn"><i class="bi bi-linkedin"></i></a>
          </div>
        </div>
      </div>
    </div>
  `;
}
