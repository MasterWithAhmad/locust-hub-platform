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

// API base URL - use the same logic as in api.js
const API_BASE_URL = window.location.origin.includes("3000")
  ? "http://localhost:5000/api"
  : `${window.location.origin}/api`;

// Initialize the blog page
document.addEventListener("DOMContentLoaded", function () {
  loadBlogPosts();
});

// Load blog posts from the API
async function loadBlogPosts() {
  try {
    console.log("Loading blog posts from:", `${API_BASE_URL}/blogposts`);

    const response = await fetch(`${API_BASE_URL}/blogposts`, {
      method: "GET",
      headers: {
        "Content-Type": "application/json",
        Accept: "application/json",
      },
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data = await response.json();
    console.log("Blog posts loaded:", data);

    allPosts = Array.isArray(data) ? data : [];
    filteredPosts = [...allPosts];

    displayPosts();
    setupPagination();
  } catch (error) {
    console.error("Error loading blog posts:", error);
    displayErrorMessage();
  }
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
    container.appendChild(postElement);
  });

  // Initialize AOS for new elements
  if (typeof AOS !== "undefined") {
    AOS.refresh();
  }
}

// Create a single post element
function createPostElement(post, index) {
  const col = document.createElement("div");
  col.className = "col-lg-4 col-md-6";
  col.setAttribute("data-aos", "fade-up");
  col.setAttribute("data-aos-delay", (index % 3) * 100);

  // Format date
  const postDate = new Date(post.date);
  const formattedDate = postDate.toLocaleDateString("en-US", {
    year: "numeric",
    month: "long",
    day: "numeric",
  });

  // Create excerpt from content (first 150 characters)
  const excerpt = post.content
    ? post.content.substring(0, 150) + "..."
    : "No content available.";

  // Determine category from tags or region
  let category = "General";
  let categoryColor = "bg-secondary";

  if (post.tags) {
    const tags = post.tags.toLowerCase();
    if (tags.includes("research")) {
      category = "Research";
      categoryColor = "bg-primary";
    } else if (tags.includes("technology")) {
      category = "Technology";
      categoryColor = "bg-success";
    } else if (tags.includes("agriculture")) {
      category = "Agriculture";
      categoryColor = "bg-warning";
    } else if (tags.includes("prediction")) {
      category = "Prediction";
      categoryColor = "bg-info";
    } else if (tags.includes("news")) {
      category = "News";
      categoryColor = "bg-danger";
    }
  } else if (post.region) {
    category = post.region;
    categoryColor = "bg-info";
  }

  // Handle post image: use provided URL or a default icon
  let imageHtml = "";
  if (post.image_url) {
    imageHtml = `
          <div class="card-img-container">
            <img src="${post.image_url}" class="card-img-top" alt="${post.title}">
          </div>
        `;
  } else {
    imageHtml = `
          <div class="card-img-top bg-gradient-primary d-flex align-items-center justify-content-center text-white" style="height: 200px;">
            <i class="bi bi-journal-text" style="font-size: 3rem;"></i>
          </div>
        `;
  }

  col.innerHTML = `
        <article class="card h-100 shadow-sm">
          ${imageHtml}
          <div class="card-body d-flex flex-column">
            <div class="mb-2">
              <span class="badge ${categoryColor}">${category}</span>
              ${
                post.region
                  ? `<span class="badge bg-outline-secondary ms-1">${post.region}</span>`
                  : ""
              }
              ${
                post.country
                  ? `<span class="badge bg-outline-secondary ms-1">${post.country}</span>`
                  : ""
              }
            </div>
            <h5 class="card-title">
              <a href="#" onclick="openPostModal(${
                post.id
              })" class="text-decoration-none text-dark">
                ${post.title}
              </a>
            </h5>
            <p class="card-text text-muted flex-grow-1">${excerpt}</p>
            <div class="card-meta d-flex justify-content-between align-items-center text-muted small mb-3">
              <span><i class="bi bi-person me-1"></i>${
                post.author || "LocustHub Team"
              }</span>
              <span><i class="bi bi-calendar me-1"></i>${formattedDate}</span>
            </div>
            <a href="#" onclick="openPostModal(${
              post.id
            })" class="btn btn-primary btn-sm align-self-start">
              Read More <i class="bi bi-arrow-right ms-1"></i>
            </a>
          </div>
        </article>
      `;

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
