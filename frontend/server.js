const express = require('express');
const path = require('path');

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Serve static files from public directory
app.use(express.static(path.join(__dirname, 'public')));

// Serve plots directory if it exists
app.use('/plots', express.static(path.join(__dirname, '../thesis-project/plots')));

// API Routes (if any) would go here

// Start server
app.listen(PORT, () => {
    console.log(`Server running on http://localhost:${PORT}`);
});