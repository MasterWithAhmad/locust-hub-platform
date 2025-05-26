const express = require('express');
const router = express.Router();
const bcrypt = require('bcrypt');
const crypto = require('crypto');
const { pool } = require('../db');
const { sendEmail } = require('../utils/email');

// ... existing routes ...

// Request password reset
router.post('/request-password-reset', async (req, res) => {
  const { email } = req.body;

  try {
    // Check if user exists
    const [users] = await pool.query('SELECT id FROM users WHERE email = ?', [email]);
    
    if (users.length === 0) {
      return res.status(404).json({ message: 'No account found with that email address' });
    }

    // Generate reset token
    const resetToken = crypto.randomBytes(32).toString('hex');
    const resetTokenExpires = new Date(Date.now() + 24 * 60 * 60 * 1000); // 24 hours from now

    // Save reset token to database
    await pool.query(
      'UPDATE users SET reset_token = ?, reset_token_expires = ?, reset_token_created_at = NOW() WHERE email = ?',
      [resetToken, resetTokenExpires, email]
    );

    // Send reset email
    const resetUrl = `${process.env.FRONTEND_URL}/reset-password.html?token=${resetToken}`;
    await sendEmail({
      to: email,
      subject: 'Password Reset Request',
      html: `
        <p>You requested a password reset.</p>
        <p>Click this <a href="${resetUrl}">link</a> to reset your password.</p>
        <p>This link will expire in 24 hours.</p>
        <p>If you didn't request this, please ignore this email.</p>
      `
    });

    res.json({ message: 'Password reset email sent' });
  } catch (error) {
    console.error('Password reset request error:', error);
    res.status(500).json({ message: 'Error processing password reset request' });
  }
});

// Reset password
router.post('/reset-password', async (req, res) => {
  const { token, newPassword } = req.body;

  try {
    // Find user with valid reset token
    const [users] = await pool.query(
      'SELECT id FROM users WHERE reset_token = ? AND reset_token_expires > NOW()',
      [token]
    );

    if (users.length === 0) {
      return res.status(400).json({ message: 'Invalid or expired reset token' });
    }

    // Hash new password
    const hashedPassword = await bcrypt.hash(newPassword, 10);

    // Update password and clear reset token
    await pool.query(
      'UPDATE users SET password = ?, reset_token = NULL, reset_token_expires = NULL, reset_token_created_at = NULL WHERE reset_token = ?',
      [hashedPassword, token]
    );

    res.json({ message: 'Password has been reset successfully' });
  } catch (error) {
    console.error('Password reset error:', error);
    res.status(500).json({ message: 'Error resetting password' });
  }
});

// Get security question
router.post('/security-question', async (req, res) => {
  const { email } = req.body;

  try {
    // Get user's security question
    const [users] = await pool.query(
      'SELECT security_question FROM users WHERE email = ?',
      [email]
    );
    
    if (users.length === 0) {
      return res.status(404).json({ message: 'No account found with that email address' });
    }

    res.json({ question: users[0].security_question });
  } catch (error) {
    console.error('Get security question error:', error);
    res.status(500).json({ message: 'Error getting security question' });
  }
});

// Verify security answer and return password
router.post('/verify-answer', async (req, res) => {
  const { email, answer } = req.body;

  try {
    // Get user's security answer and password
    const [users] = await pool.query(
      'SELECT security_answer, password FROM users WHERE email = ?',
      [email]
    );
    
    if (users.length === 0) {
      return res.status(404).json({ message: 'No account found with that email address' });
    }

    // Compare answers (case-insensitive)
    if (users[0].security_answer.toLowerCase() !== answer.toLowerCase()) {
      return res.status(400).json({ message: 'Incorrect security answer' });
    }

    res.json({ password: users[0].password });
  } catch (error) {
    console.error('Verify security answer error:', error);
    res.status(500).json({ message: 'Error verifying security answer' });
  }
});

module.exports = router; 