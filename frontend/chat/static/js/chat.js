document.addEventListener('DOMContentLoaded', function() {
    const sidebar = document.getElementById('sidebar');
    let timeout;

    // Sidebar toggle functionality
    function handleSidebar(e) {
        if (window.innerWidth <= 768) {  // Mobile view
            if (e.clientX <= 20) {  // Show sidebar
                clearTimeout(timeout);
                sidebar.style.transform = 'translateX(0)';
            } else if (e.clientX > 250) {  // Hide sidebar
                timeout = setTimeout(() => {
                    sidebar.style.transform = 'translateX(-250px)';
                }, 300);
            }
        }
    }

    // Add mouse move event listener for sidebar
    document.addEventListener('mousemove', handleSidebar);

    // Handle touch events for mobile
    let touchStartX = 0;
    let touchEndX = 0;

    document.addEventListener('touchstart', (e) => {
        touchStartX = e.changedTouches[0].screenX;
    });

    document.addEventListener('touchend', (e) => {
        touchEndX = e.changedTouches[0].screenX;
        handleSwipe();
    });

    function handleSwipe() {
        const swipeDistance = touchEndX - touchStartX;

        if (swipeDistance > 50) {  // Right swipe
            sidebar.style.transform = 'translateX(0)';
        } else if (swipeDistance < -50) {  // Left swipe
            sidebar.style.transform = 'translateX(-250px)';
        }
    }

    // Auto-scroll to bottom of messages
    const messagesContainer = document.querySelector('.messages-container');
    if (messagesContainer) {
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }

    // Form submission handling
    const messageForm = document.querySelector('.message-form');
    if (messageForm) {
        messageForm.addEventListener('submit', function() {
            const button = this.querySelector('button');
            button.disabled = true;
            button.textContent = 'Sending...';
        });
    }

    // Highlight active chat
    const currentChatId = new URLSearchParams(window.location.search).get('chat_id');
    if (currentChatId) {
        const activeChat = document.querySelector(`.chat-item a[href$="chat_id=${currentChatId}"]`);
        if (activeChat) {
            activeChat.parentElement.classList.add('active');
        }
    }

    // Handle message input
    const messageInput = document.querySelector('.message-form input[name="message"]');
    if (messageInput) {
        // Enable submit button only when input has text
        const submitButton = document.querySelector('.message-form button');
        messageInput.addEventListener('input', function() {
            submitButton.disabled = !this.value.trim();
        });

        // Submit on Enter (but allow Shift+Enter for new line)
        messageInput.addEventListener('keydown', function(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                if (this.value.trim()) {
                    messageForm.submit();
                }
            }
        });
    }

    // Periodic check for new messages
    let lastMessageTimestamp = null;
    if (messagesContainer) {
        const messages = messagesContainer.querySelectorAll('.message');
        if (messages.length > 0) {
            lastMessageTimestamp = messages[messages.length - 1].dataset.timestamp;
        }
    }
});