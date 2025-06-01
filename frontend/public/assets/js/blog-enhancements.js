/**
 * Blog Enhancements for LocustHub
 * Implements: Infinite scroll, reading time, table of contents, save for later, social sharing, and more
 */

document.addEventListener('DOMContentLoaded', function() {
    // Initialize variables
    let currentPage = 1;
    let isLoading = false;
    let hasMorePosts = true;
    const postsPerPage = 6;
    let savedPosts = JSON.parse(localStorage.getItem('savedPosts')) || [];
    
    // Initialize components
    initInfiniteScroll();
    initSocialSharing();
    initSaveForLater();
    initImageZoom();
    initSmoothScroll();
    initAccessibility();
    
    // Initialize tooltips
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
    
    // Calculate reading time for posts
    function calculateReadingTime(content) {
        const wordsPerMinute = 200;
        const text = content.textContent || content.innerText;
        const wordCount = text.trim().split(/\s+/).length;
        return Math.ceil(wordCount / wordsPerMinute);
    }
    
    // Generate table of contents
    function generateTOC(container, contentSelector, mobile = false) {
        const content = document.querySelector(contentSelector);
        if (!content) return;
        
        const headings = content.querySelectorAll('h2, h3, h4');
        if (headings.length === 0) {
            container.style.display = 'none';
            return;
        }
        
        let tocHtml = '<ul class="list-unstyled">';
        headings.forEach((heading, index) => {
            const level = parseInt(heading.tagName.substring(1));
            const indent = level > 2 ? ' ms-3' : '';
            const id = `heading-${index}`;
            heading.id = id;
            
            tocHtml += `
                <li class="mb-2${indent}">
                    <a href="#${id}" class="text-decoration-none" data-bs-toggle="${mobile ? 'collapse' : ''}">
                        ${heading.textContent}
                    </a>
                </li>`;
        });
        tocHtml += '</ul>';
        
        container.innerHTML = tocHtml;
    }
    
    // Initialize infinite scroll
    function initInfiniteScroll() {
        const blogContainer = document.querySelector('.blog-container');
        if (!blogContainer) return;
        
        window.addEventListener('scroll', () => {
            if (isLoading || !hasMorePosts) return;
            
            const { scrollTop, scrollHeight, clientHeight } = document.documentElement;
            
            if (scrollTop + clientHeight >= scrollHeight - 300) {
                loadMorePosts();
            }
        });
    }
    
    // Load more posts with AJAX
    async function loadMorePosts() {
        if (isLoading || !hasMorePosts) return;
        
        isLoading = true;
        const loadingIndicator = document.getElementById('loadingIndicator');
        if (loadingIndicator) loadingIndicator.classList.remove('d-none');
        
        try {
            const response = await fetch(`${window.API_BASE_URL || ''}/api/blogposts?page=${currentPage + 1}&limit=${postsPerPage}`, {
                headers: {
                    'Accept': 'application/json',
                    'Content-Type': 'application/json'
                }
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            // Check if response is JSON
            const contentType = response.headers.get('content-type');
            if (!contentType || !contentType.includes('application/json')) {
                throw new Error('Received non-JSON response from server');
            }
            
            const data = await response.json();
            const posts = Array.isArray(data) ? data : [];
            
            if (posts.length === 0 || posts.length < postsPerPage) {
                hasMorePosts = false;
                const noMorePosts = document.getElementById('noMorePosts');
                if (noMorePosts) noMorePosts.classList.remove('d-none');
                return;
            }
            
            currentPage++;
            const fragment = document.createDocumentFragment();
            
            posts.forEach(post => {
                if (post && post.id) {  // Only process valid posts
                    const postElement = createPostElement(post);
                    if (postElement) fragment.appendChild(postElement);
                }
            });
            
            const blogContainer = document.querySelector('.blog-container');
            if (blogContainer && fragment.hasChildNodes()) {
                blogContainer.appendChild(fragment);
                
                // Re-initialize tooltips for new elements
                const newTooltips = [].slice.call(fragment.querySelectorAll('[data-bs-toggle="tooltip"]'));
                newTooltips.forEach(el => new bootstrap.Tooltip(el));
                
                // Initialize AOS for new elements
                if (typeof AOS !== 'undefined') {
                    AOS.refresh();
                }
            }
            
        } catch (error) {
            console.error('Error loading more posts:', error);
            // Show error message to user
            const errorElement = document.createElement('div');
            errorElement.className = 'alert alert-warning mt-3';
            errorElement.textContent = 'Failed to load more posts. Please try again later.';
            document.querySelector('.blog-container')?.appendChild(errorElement);
            
            // Only disable loading if it's a client-side error
            if (error.message.includes('Failed to fetch')) {
                hasMorePosts = false;
            }
        } finally {
            isLoading = false;
            if (loadingIndicator) loadingIndicator.classList.add('d-none');
        }
    }
    
    // Initialize social sharing
    function initSocialSharing() {
        document.addEventListener('click', (e) => {
            const postUrl = encodeURIComponent(window.location.href);
            const postTitle = encodeURIComponent(document.title);
            
            if (e.target.closest('.share-facebook')) {
                e.preventDefault();
                window.open(`https://www.facebook.com/sharer/sharer.php?u=${postUrl}`, 'facebook-share-dialog', 'width=626,height=436');
            } else if (e.target.closest('.share-twitter')) {
                e.preventDefault();
                window.open(`https://twitter.com/intent/tweet?url=${postUrl}&text=${postTitle}`, 'twitter-share', 'width=550,height=235');
            } else if (e.target.closest('.share-linkedin')) {
                e.preventDefault();
                window.open(`https://www.linkedin.com/sharing/share-offsite/?url=${postUrl}`, 'linkedin-share', 'width=550,height=400');
            } else if (e.target.closest('.share-copy')) {
                e.preventDefault();
                navigator.clipboard.writeText(window.location.href).then(() => {
                    const tooltip = bootstrap.Tooltip.getInstance(e.target.closest('.share-copy'));
                    const originalTitle = e.target.closest('.share-copy').getAttribute('data-bs-original-title');
                    e.target.closest('.share-copy').setAttribute('data-bs-original-title', 'Link copied!');
                    tooltip.show();
                    
                    setTimeout(() => {
                        e.target.closest('.share-copy').setAttribute('data-bs-original-title', originalTitle);
                        tooltip.hide();
                    }, 2000);
                });
            }
        });
    }
    
    // Initialize save for later functionality
    function initSaveForLater() {
        document.addEventListener('click', (e) => {
            if (e.target.closest('#save-post-btn')) {
                e.preventDefault();
                const postId = e.target.closest('.blog-post-card').dataset.postId;
                toggleSavePost(postId, e.target.closest('#save-post-btn'));
            }
        });
        
        // Update saved posts buttons on page load
        updateSavedPostsButtons();
    }
    
    function toggleSavePost(postId, button) {
        const index = savedPosts.indexOf(postId);
        
        if (index === -1) {
            savedPosts.push(postId);
            if (button) {
                button.innerHTML = '<i class="fas fa-bookmark me-1"></i> Saved';
                button.classList.add('active');
            }
        } else {
            savedPosts.splice(index, 1);
            if (button) {
                button.innerHTML = '<i class="far fa-bookmark me-1"></i> Save for later';
                button.classList.remove('active');
            }
        }
        
        localStorage.setItem('savedPosts', JSON.stringify(savedPosts));
    }
    
    function updateSavedPostsButtons() {
        document.querySelectorAll('.blog-post-card').forEach(card => {
            const postId = card.dataset.postId;
            const saveButton = card.querySelector('.save-post-btn');
            
            if (saveButton && savedPosts.includes(postId)) {
                saveButton.innerHTML = '<i class="fas fa-bookmark me-1"></i> Saved';
                saveButton.classList.add('active');
            }
        });
    }
    
    // Initialize image zoom functionality
    function initImageZoom() {
        if (typeof lightGallery !== 'undefined') {
            lightGallery(document.querySelector('.blog-post-content'), {
                selector: '.zoomable-image',
                download: false,
                zoom: true,
                zoomFromOrigin: false,
                zoomScale: 2,
                actualSize: false
            });
        }
        
        // Add zoom class to images
        document.querySelectorAll('.blog-post-content img').forEach(img => {
            img.classList.add('zoomable-image');
            img.style.cursor = 'zoom-in';
        });
    }
    
    // Initialize smooth scrolling
    function initSmoothScroll() {
        document.querySelectorAll('a[href^="#"]').forEach(anchor => {
            anchor.addEventListener('click', function(e) {
                e.preventDefault();
                
                const targetId = this.getAttribute('href');
                if (targetId === '#') return;
                
                const targetElement = document.querySelector(targetId);
                if (targetElement) {
                    targetElement.scrollIntoView({
                        behavior: 'smooth',
                        block: 'start'
                    });
                    
                    // Update URL without page reload
                    if (history.pushState) {
                        history.pushState(null, null, targetId);
                    } else {
                        location.hash = targetId;
                    }
                }
            });
        });
    }
    
    // Initialize accessibility features
    function initAccessibility() {
        // Add skip to main content link
        const skipLink = document.createElement('a');
        skipLink.href = '#main-content';
        skipLink.className = 'skip-to-content';
        skipLink.textContent = 'Skip to main content';
        document.body.insertBefore(skipLink, document.body.firstChild);
        
        // Add ARIA labels to interactive elements
        document.querySelectorAll('button, a, input, select, textarea, [tabindex="0"]').forEach(el => {
            if (!el.hasAttribute('aria-label') && !el.textContent.trim()) {
                el.setAttribute('aria-label', el.getAttribute('title') || 'Interactive element');
            }
        });
        
        // Ensure proper contrast for text
        document.querySelectorAll('p, h1, h2, h3, h4, h5, h6, a, span, div').forEach(el => {
            const style = window.getComputedStyle(el);
            const bgColor = style.backgroundColor;
            const textColor = style.color;
            // You could add contrast checking logic here
        });
    }
    
    // Initialize Disqus comments
    function loadDisqus() {
        if (typeof DISQUS !== 'undefined') {
            DISQUS.reset({
                reload: true,
                config: function() {
                    this.page.url = window.location.href;
                    this.page.identifier = window.location.pathname;
                }
            });
        }
    }
    
    // Event delegation for dynamic content
    document.addEventListener('click', (e) => {
        // Handle clicks on post cards
        if (e.target.closest('.blog-post-card')) {
            const card = e.target.closest('.blog-post-card');
            if (!e.target.closest('a, button, [role="button"]')) {
                window.location.href = card.dataset.postUrl || '#';
            }
        }
    });
    
    // Initialize Intersection Observer for lazy loading
    if ('IntersectionObserver' in window) {
        const lazyImages = document.querySelectorAll('img.lazy-load');
        const imageObserver = new IntersectionObserver((entries, observer) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    const img = entry.target;
                    img.src = img.dataset.src;
                    img.classList.remove('lazy-load');
                    observer.unobserve(img);
                }
            });
        });
        
        lazyImages.forEach(img => imageObserver.observe(img));
    }
    
    // Initialize tooltips
    const tooltripTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltripTriggerList.map(function (tooltripTriggerEl) {
        return new bootstrap.Tooltip(tooltripTriggerEl);
    });
    
    // Initialize popovers
    const popoverTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="popover"]'));
    popoverTriggerList.map(function (popoverTriggerEl) {
        return new bootstrap.Popover(popoverTriggerEl);
    });
    
    // Handle back to top button
    const backToTopButton = document.querySelector('.back-to-top');
    if (backToTopButton) {
        window.addEventListener('scroll', () => {
            if (window.pageYOffset > 300) {
                backToTopButton.classList.add('show');
            } else {
                backToTopButton.classList.remove('show');
            }
        });
        
        backToTopButton.addEventListener('click', (e) => {
            e.preventDefault();
            window.scrollTo({
                top: 0,
                behavior: 'smooth'
            });
        });
    }
});

// Helper function to debounce function calls
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// Helper function to throttle function calls
function throttle(func, limit) {
    let inThrottle;
    return function() {
        const args = arguments;
        const context = this;
        if (!inThrottle) {
            func.apply(context, args);
            inThrottle = true;
            setTimeout(() => inThrottle = false, limit);
        }
    };
}
