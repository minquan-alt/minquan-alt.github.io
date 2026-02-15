// ===========================
// Load and Display Project Details
// ===========================
function loadProjectDetail() {
    // Get project ID from URL
    const urlPath = window.location.pathname;
    const projectId = urlPath.split('/').pop();
    
    const project = projectsData[projectId];
    
    if (!project) {
        document.getElementById('project-detail-container').innerHTML = `
            <div style="text-align: center; padding: 4rem 0;">
                <h1>Project Not Found</h1>
                <p style="color: var(--text-secondary); margin: 1rem 0;">The project you're looking for doesn't exist.</p>
                <a href="/" class="btn btn-primary" style="display: inline-flex; margin-top: 1rem;">Back to Home</a>
            </div>
        `;
        return;
    }
    
    // Build the project detail HTML
    const detailHTML = `
        <div class="project-header">
            <h1 class="project-title">${project.title}</h1>
            <p class="project-subtitle">${project.subtitle}</p>
            
            <div class="project-meta-info">
                <div class="meta-item">
                    <i class="fas fa-calendar"></i>
                    <span>2025</span>
                </div>
                <div class="meta-item">
                    <i class="fas fa-users"></i>
                    <span>${project.teamSize} ${project.teamSize === 1 ? 'member' : 'members'}</span>
                </div>
                <div class="meta-item">
                    <i class="fas fa-user-tag"></i>
                    <span>${project.role}</span>
                </div>
            </div>
            
            <div class="project-tags" style="margin-top: 1rem;">
                ${project.tags.map(tag => `<span class="tag">${tag}</span>`).join('')}
            </div>
        </div>
        
        <!-- Problem Section -->
        <section class="detail-section">
            <h2>
                <i class="fas ${project.problem.icon}"></i>
                ${project.problem.title}
            </h2>
            ${project.problem.content}
        </section>
        
        <!-- Data Section -->
        <section class="detail-section">
            <h2>
                <i class="fas ${project.data.icon}"></i>
                ${project.data.title}
            </h2>
            ${project.data.content}
        </section>
        
        <!-- Architecture Section -->
        <section class="detail-section">
            <h2>
                <i class="fas ${project.architecture.icon}"></i>
                ${project.architecture.title}
            </h2>
            ${project.architecture.content}
        </section>
        
        <!-- Experiments Section -->
        <section class="detail-section">
            <h2>
                <i class="fas ${project.experiments.icon}"></i>
                ${project.experiments.title}
            </h2>
            ${project.experiments.content}
        </section>
        
        <!-- Deployment Section -->
        <section class="detail-section">
            <h2>
                <i class="fas ${project.deployment.icon}"></i>
                ${project.deployment.title}
            </h2>
            ${project.deployment.content}
        </section>
    `;
    
    document.getElementById('project-detail-container').innerHTML = detailHTML;
    
    // Update page title
    document.title = `${project.title} - Quang Minh Portfolio`;
    
    // Add scroll reveal animation
    setTimeout(() => {
        const sections = document.querySelectorAll('.detail-section');
        sections.forEach((section, index) => {
            section.style.opacity = '0';
            section.style.transform = 'translateY(30px)';
            setTimeout(() => {
                section.style.transition = 'all 0.6s ease';
                section.style.opacity = '1';
                section.style.transform = 'translateY(0)';
            }, index * 100);
        });
    }, 100);
}

// ===========================
// Initialize on Page Load
// ===========================
document.addEventListener('DOMContentLoaded', () => {
    loadProjectDetail();
});

// ===========================
// Smooth Scroll for Links
// ===========================
document.addEventListener('click', (e) => {
    if (e.target.matches('a[href^="#"]')) {
        e.preventDefault();
        const target = document.querySelector(e.target.getAttribute('href'));
        if (target) {
            target.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
    }
});
