const express = require('express');
const path = require('path');
const app = express();

// Serve static files from the public directory
app.use(express.static(path.join(__dirname, 'public')));

// Serve plots directory
app.use('/plots', express.static(path.join(__dirname, '../thesis-project/plots')));

// Routes
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

app.get('/about', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'about.html'));
});

app.get('/contact', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'contact.html'));
});

app.get('/predict', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'predict.html'));
});

// Prediction endpoint
app.post('/predict', express.json(), (req, res) => {
    // Your prediction logic here
    const { age, income, education, experience } = req.body;
    
    // Mock prediction response
    const prediction = {
        result: "Based on the provided data, the predicted salary range is $80,000 - $95,000",
        confidence: 85
    };
    
    res.json(prediction);
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
    console.log(`Server is running on port ${PORT}`);
}); 