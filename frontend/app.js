const express = require('express');
const path = require('path');

const app = express();
const port = process.env.PORT || 3000;

// Serve static files from the 'public' directory
app.use(express.static(path.join(__dirname, 'public')));

// Define routes for all pages
app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

app.get('/services-details', (req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'services-details.html'));
});

app.get('/portfolio-details', (req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'portfolio-details.html'));
});

app.get('/starter-page', (req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'starter-page.html'));
});

// Define a route to handle prediction requests
app.post('/predict', (req, res) => {
  const { input1, input2 } = req.body;

  // TODO: Send data to Flask backend and get prediction
  // For now, return a dummy prediction
  const prediction = `Prediction for ${input1} and ${input2}: 0.5`;

  res.json({ prediction });
});

// Start the server
app.listen(port, () => {
  console.log(`Server is running at http://localhost:${port}`);
}); 