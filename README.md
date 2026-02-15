# 🚀 Quang Minh - AI Engineer Portfolio

Modern, professional portfolio website built with Node.js, showcasing AI/ML projects with a focus on system thinking and deployment.

## ✨ Features

- **Dark Mode Design** - Professional tech vibe with black/gray/blue color scheme
- **Responsive Layout** - Works seamlessly on desktop, tablet, and mobile
- **Smooth Animations** - Fade-in effects and scroll reveal animations
- **Project Showcase** - Detailed project pages with problem, data, architecture, experiments, and deployment sections
- **System Thinking Section** - Demonstrates engineering mindset beyond just Kaggle competitions
- **Technical Blog** - Space for sharing technical insights and learnings
- **Fast & Lightweight** - Pure HTML/CSS/JS frontend with Express.js backend

## 🛠️ Tech Stack

- **Backend**: Node.js + Express.js
- **Frontend**: HTML5, CSS3 (CSS Variables), Vanilla JavaScript
- **Fonts**: Inter, Space Grotesk, JetBrains Mono (Google Fonts)
- **Icons**: Font Awesome 6
- **Deployment**: Can be deployed to Vercel, Heroku, or any Node.js hosting

## 📦 Installation

### Prerequisites

- Node.js (v14 or higher)
- npm or yarn

### Setup

1. Clone or navigate to the project directory:
```bash
cd portfolio
```

2. Install dependencies:
```bash
npm install
```

3. Start the development server:
```bash
npm run dev
```

4. Open your browser and visit:
```
http://localhost:3000
```

## 🚀 Production Build

To run in production mode:

```bash
npm start
```

The server will start on port 3000 (or the PORT environment variable if set).

## 📁 Project Structure

```
portfolio/
├── public/
│   ├── css/
│   │   ├── style.css           # Main stylesheet
│   │   └── project-detail.css  # Project detail page styles
│   ├── js/
│   │   ├── main.js             # Main JavaScript with animations
│   │   ├── project-data.js     # Project data and content
│   │   └── project-detail.js   # Project detail page logic
│   ├── images/                 # Project images and screenshots
│   ├── index.html              # Main landing page
│   └── project-detail.html     # Project detail page template
├── server.js                   # Express server
├── package.json
└── README.md
```

## 🎨 Customization

### Update Personal Information

1. **Hero Section**: Edit `public/index.html` - Update name, title, description
2. **About Me**: Edit the about section content in `index.html`
3. **Tech Stack**: Modify the tech tags in the about section
4. **Social Links**: Update GitHub, LinkedIn, and email links

### Add New Projects

1. Open `public/js/project-data.js`
2. Add a new project object with the following structure:

```javascript
4: {
    id: 4,
    title: "Your Project Title",
    subtitle: "Short description",
    tags: ["Tech1", "Tech2", "Tech3"],
    github: "https://github.com/username/repo",
    demo: "https://demo-url.com",
    
    problem: { /* Problem description */ },
    data: { /* Data information */ },
    architecture: { /* Architecture details */ },
    experiments: { /* Experiments and results */ },
    deployment: { /* Deployment information */ }
}
```

3. Add a corresponding card in `index.html` in the projects section

### Update Images

Place your project images in the `public/images/` folder:
- `project1.jpg` - Object Detection System
- `project2.jpg` - RAG Chatbot
- `project3.jpg` - Demand Forecasting

Recommended image size: 800x500px (16:10 ratio)

### Customize Colors

Edit CSS variables in `public/css/style.css`:

```css
:root {
    --bg-primary: #0a0a0a;      /* Main background */
    --accent-primary: #3b82f6;   /* Accent color (blue) */
    --text-primary: #e0e0e0;     /* Main text color */
    /* ... other variables */
}
```

## 📝 Blog Posts

To add blog content:

1. Create individual blog post HTML files in `public/blog/`
2. Update the blog card links in `index.html`
3. Or integrate with a CMS/Markdown system

## 🌐 Deployment

### Vercel (Recommended)

1. Install Vercel CLI:
```bash
npm install -g vercel
```

2. Deploy:
```bash
vercel
```

### Heroku

1. Create a `Procfile`:
```
web: node server.js
```

2. Deploy:
```bash
heroku create
git push heroku main
```

### Traditional Hosting

1. Upload files to your server
2. Install Node.js on the server
3. Run `npm install` and `npm start`
4. Use PM2 for process management:
```bash
npm install -g pm2
pm2 start server.js --name portfolio
pm2 save
pm2 startup
```

## 🔧 Environment Variables

```bash
PORT=3000  # Server port (default: 3000)
```

## 📄 License

MIT License - feel free to use this template for your own portfolio!

## 🤝 Contributing

This is a personal portfolio, but feel free to fork and customize for your own use.

## 📬 Contact

- **Email**: quangminh@email.com
- **GitHub**: [@quangminh](https://github.com/quangminh)
- **LinkedIn**: [quangminh](https://linkedin.com/in/quangminh)

---

**Built with ❤️ by Quang Minh** | February 2026
