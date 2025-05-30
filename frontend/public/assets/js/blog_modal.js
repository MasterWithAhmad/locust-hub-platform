// Blog Modal Logic for Enhanced Blog Creation
// Requires Quill.js and SweetAlert2

// Load Quill dynamically if not present
function loadQuillIfNeeded(callback) {
  if (window.Quill) {
    callback();
    return;
  }
  var link = document.createElement('link');
  link.rel = 'stylesheet';
  link.href = 'https://cdn.quilljs.com/1.3.6/quill.snow.css';
  document.head.appendChild(link);
  var script = document.createElement('script');
  script.src = 'https://cdn.quilljs.com/1.3.6/quill.min.js';
  script.onload = callback;
  document.body.appendChild(script);
}

// Show the enhanced blog modal
function showBlogModal({
  onPublish,
  onPreview,
  draftKey = 'blogDraft',
  user,
  region,
  country,
  imageFilePersist
}) {
  loadQuillIfNeeded(() => {
    let imageFileState = imageFilePersist || null;
    // Modal HTML
    const modalHtml = `
      <form id="blog-form" autocomplete="off" style="text-align:left;">
        <div class="mb-3">
          <label for="blog-title" class="form-label">Title <span id="title-count" class="text-muted small">0/80</span></label>
          <input id="blog-title" class="form-control" maxlength="80" required placeholder="Enter a title...">
        </div>
        <div class="mb-3">
          <label for="blog-tags" class="form-label">Tags <span class="text-muted small">(comma separated)</span></label>
          <input id="blog-tags" class="form-control" maxlength="60" placeholder="e.g. Event, Alert, Experience">
        </div>
        <div class="mb-3">
          <label class="form-label">Content <span id="content-count" class="text-muted small">0/2000</span></label>
          <div id="quill-editor" style="height:180px;"></div>
        </div>
        <div class="mb-3">
          <label class="form-label">Image (optional)</label>
          <input type="file" id="blog-image" accept="image/*" class="form-control">
          <div id="image-preview" class="mt-2"></div>
        </div>
      </form>
    `;
    Swal.fire({
      title: 'New Event/Blog Post',
      html: modalHtml,
      width: 700,
      showCancelButton: true,
      showDenyButton: true,
      confirmButtonText: 'Publish',
      denyButtonText: 'Preview',
      cancelButtonText: 'Cancel',
      focusConfirm: false,
      didOpen: () => {
        // Quill setup
        const quill = new Quill('#quill-editor', {
          theme: 'snow',
          placeholder: 'Write your event or blog post here...'
        });
        // Restore draft if present
        let draft = localStorage.getItem(draftKey);
        if (draft) {
          try {
            draft = JSON.parse(draft);
            document.getElementById('blog-title').value = draft.title || '';
            document.getElementById('blog-tags').value = draft.tags || '';
            quill.root.innerHTML = draft.content || '';
          } catch {}
        }
        // Live counters
        const titleInput = document.getElementById('blog-title');
        const contentCount = document.getElementById('content-count');
        const titleCount = document.getElementById('title-count');
        function updateCounts() {
          titleCount.textContent = `${titleInput.value.length}/80`;
          const plain = quill.getText().trim();
          contentCount.textContent = `${plain.length}/2000`;
        }
        titleInput.addEventListener('input', updateCounts);
        quill.on('text-change', updateCounts);
        updateCounts();
        // Autosave draft
        function saveDraft() {
          localStorage.setItem(draftKey, JSON.stringify({
            title: titleInput.value,
            tags: document.getElementById('blog-tags').value,
            content: quill.root.innerHTML
          }));
        }
        titleInput.addEventListener('input', saveDraft);
        document.getElementById('blog-tags').addEventListener('input', saveDraft);
        quill.on('text-change', saveDraft);
        // Image preview and persistence
        const imageInput = document.getElementById('blog-image');
        const preview = document.getElementById('image-preview');
        if (imageFileState) {
          const reader = new FileReader();
          reader.onload = function(evt) {
            preview.innerHTML = `<img src="${evt.target.result}" alt="Preview" style="max-width:100%;max-height:120px;border-radius:8px;">`;
          };
          reader.readAsDataURL(imageFileState);
        }
        imageInput.addEventListener('change', function(e) {
          imageFileState = e.target.files[0];
          if (imageFileState) {
            const reader = new FileReader();
            reader.onload = function(evt) {
              preview.innerHTML = `<img src="${evt.target.result}" alt="Preview" style="max-width:100%;max-height:120px;border-radius:8px;">`;
            };
            reader.readAsDataURL(imageFileState);
          } else {
            preview.innerHTML = '';
          }
        });
      },
      preConfirm: () => {
        const title = document.getElementById('blog-title').value.trim();
        const tags = document.getElementById('blog-tags').value.trim();
        const quill = Quill.find(document.getElementById('quill-editor'));
        const content = quill.root.innerHTML.trim();
        const plain = quill.getText().trim();
        if (!title || !plain) {
          Swal.showValidationMessage('Please enter both a title and content.');
          return false;
        }
        if (title.length > 80) {
          Swal.showValidationMessage('Title is too long.');
          return false;
        }
        if (plain.length > 2000) {
          Swal.showValidationMessage('Content is too long.');
          return false;
        }
        // Image
        return { title, tags, content, imageFile: imageFileState };
      }
    }).then(async (result) => {
      if (result.isDenied) {
        // Preview
        const title = document.getElementById('blog-title').value.trim();
        const tags = document.getElementById('blog-tags').value.trim();
        const quill = Quill.find(document.getElementById('quill-editor'));
        const content = quill.root.innerHTML.trim();
        let imageHtml = '';
        if (imageFileState) {
          const reader = new FileReader();
          reader.onload = function(evt) {
            imageHtml = `<img src="${evt.target.result}" alt="Preview" style="max-width:100%;max-height:180px;border-radius:8px;">`;
            Swal.fire({
              title: title,
              html: `<div>${imageHtml}</div><div class='mt-3'>${content}</div><div class='mt-2 text-muted small'>Tags: ${tags}</div>`,
              showCancelButton: true,
              confirmButtonText: 'Back',
              cancelButtonText: 'Close'
            }).then(() => showBlogModal({ onPublish, onPreview, draftKey, user, region, country, imageFilePersist: imageFileState }));
          };
          reader.readAsDataURL(imageFileState);
        } else {
          Swal.fire({
            title: title,
            html: `<div>${content}</div><div class='mt-2 text-muted small'>Tags: ${tags}</div>`,
            showCancelButton: true,
            confirmButtonText: 'Back',
            cancelButtonText: 'Close'
          }).then(() => showBlogModal({ onPublish, onPreview, draftKey, user, region, country, imageFilePersist: null }));
        }
      } else if (result.isConfirmed && result.value) {
        // Publish
        onPublish(result.value);
        localStorage.removeItem(draftKey);
      }
    });
  });
}
