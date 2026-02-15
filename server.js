const express = require('express');
const path = require('path');

const app = express();
const PORT = process.env.PORT || 3000;

// Serve static files
app.use(express.static('public'));
app.use('/css', express.static(path.join(__dirname, 'public/css')));
app.use('/js', express.static(path.join(__dirname, 'public/js')));
app.use('/images', express.static(path.join(__dirname, 'public/images')));

// Routes
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

app.get('/project/:id', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'project-detail.html'));
});

app.listen(PORT, () => {
    console.log(`🚀 Portfolio server running at http://localhost:${PORT}`);
});
