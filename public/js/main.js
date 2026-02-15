// ===========================
// Scroll Reveal Animation
// ===========================
function revealOnScroll() {
    const reveals = document.querySelectorAll('.reveal');
    
    reveals.forEach(element => {
        const windowHeight = window.innerHeight;
        const elementTop = element.getBoundingClientRect().top;
        const elementVisible = 150;
        
        if (elementTop < windowHeight - elementVisible) {
            element.classList.add('active');
        }
    });
}

// ===========================
// Navbar Scroll Effect
// ===========================
function handleNavbarScroll() {
    const navbar = document.querySelector('.navbar');
    
    if (window.scrollY > 100) {
        navbar.style.background = 'rgba(10, 10, 10, 0.95)';
    } else {
        navbar.style.background = 'rgba(10, 10, 10, 0.8)';
    }
}

// ===========================
// Smooth Scroll for Navigation Links
// ===========================
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        
        if (target) {
            const offsetTop = target.offsetTop - 80;
            window.scrollTo({
                top: offsetTop,
                behavior: 'smooth'
            });
        }
    });
});

// ===========================
// Project View Detail
// ===========================
function viewProject(projectId, event) {
    event.preventDefault();
    window.location.href = `/project/${projectId}`;
}

// ===========================
// Add Reveal Class to Elements
// ===========================
function initRevealElements() {
    const elementsToReveal = [
        '.project-card',
        '.process-step',
        '.blog-card',
        '.tech-category'
    ];
    
    elementsToReveal.forEach(selector => {
        document.querySelectorAll(selector).forEach(element => {
            element.classList.add('reveal');
        });
    });
}

// ===========================
// Typing Effect for Hero (Optional)
// ===========================
function typeEffect(element, text, speed = 100) {
    let i = 0;
    element.textContent = '';
    
    function type() {
        if (i < text.length) {
            element.textContent += text.charAt(i);
            i++;
            setTimeout(type, speed);
        }
    }
    
    type();
}

// ===========================
// Initialize Particles Background (Optional Light Effect)
// ===========================
function createParticles() {
    const hero = document.querySelector('.hero');
    if (!hero) return;
    
    for (let i = 0; i < 50; i++) {
        const particle = document.createElement('div');
        particle.style.position = 'absolute';
        particle.style.width = '2px';
        particle.style.height = '2px';
        particle.style.background = 'rgba(59, 130, 246, 0.3)';
        particle.style.borderRadius = '50%';
        particle.style.left = Math.random() * 100 + '%';
        particle.style.top = Math.random() * 100 + '%';
        particle.style.animation = `float ${5 + Math.random() * 10}s ease-in-out infinite`;
        particle.style.animationDelay = Math.random() * 5 + 's';
        hero.appendChild(particle);
    }
}

// Add float animation
const style = document.createElement('style');
style.textContent = `
    @keyframes float {
        0%, 100% {
            transform: translate(0, 0);
            opacity: 0;
        }
        50% {
            opacity: 1;
        }
        100% {
            transform: translate(${Math.random() * 100 - 50}px, ${Math.random() * 100 - 50}px);
        }
    }
`;
document.head.appendChild(style);

// ===========================
// Event Listeners
// ===========================
window.addEventListener('scroll', () => {
    revealOnScroll();
    handleNavbarScroll();
});

window.addEventListener('load', () => {
    initRevealElements();
    revealOnScroll();
    // createParticles(); // Uncomment for particle effect
});

// ===========================
// Tech Stack Hover Effect
// ===========================
document.querySelectorAll('.tech-tag').forEach(tag => {
    tag.addEventListener('mouseenter', function() {
        this.style.background = 'rgba(59, 130, 246, 0.1)';
    });
    
    tag.addEventListener('mouseleave', function() {
        this.style.background = 'var(--bg-tertiary)';
    });
});

// ===========================
// Project Card Click Analytics (Optional)
// ===========================
document.querySelectorAll('.project-card').forEach(card => {
    card.addEventListener('click', function(e) {
        if (!e.target.classList.contains('view-detail-btn')) {
            const projectId = this.getAttribute('data-project');
            viewProject(projectId, e);
        }
    });
});

// ===========================
// Add Active State to Nav Links
// ===========================
function updateActiveNavLink() {
    const sections = document.querySelectorAll('section[id]');
    const navLinks = document.querySelectorAll('.nav-links a');
    
    let current = '';
    
    sections.forEach(section => {
        const sectionTop = section.offsetTop;
        const sectionHeight = section.clientHeight;
        
        if (window.scrollY >= sectionTop - 100) {
            current = section.getAttribute('id');
        }
    });
    
    navLinks.forEach(link => {
        link.classList.remove('active');
        if (link.getAttribute('href') === `#${current}`) {
            link.classList.add('active');
        }
    });
}

window.addEventListener('scroll', updateActiveNavLink);

// ===========================
// Console Easter Egg
// ===========================
console.log('%c👋 Hello, curious developer!', 'font-size: 20px; color: #3b82f6; font-weight: bold;');
console.log('%cInterested in the code? Check out the GitHub repo!', 'font-size: 14px; color: #a0a0a0;');
console.log('%chttps://github.com/quangminh', 'font-size: 14px; color: #3b82f6;');
