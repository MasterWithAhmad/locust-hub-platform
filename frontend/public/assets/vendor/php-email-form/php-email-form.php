<?php
/**
 * PHP Email Form Class
 * A simple class to handle email form submissions using SendGrid
 */
require '../sendgrid/sendgrid-php.php';

class PHP_Email_Form {
    public $to;
    public $from_name;
    public $from_email;
    public $subject;
    public $ajax = false;
    public $smtp = array();
    private $messages = array();

    public function add_message($message, $label = '', $min_length = 0) {
        if (!empty($message)) {
            $this->messages[] = array(
                'label' => $label,
                'message' => $message,
                'min_length' => $min_length
            );
        }
    }

    public function send() {
        // Validate required fields
        if (empty($this->to) || empty($this->from_name) || empty($this->from_email)) {
            return 'Required fields are missing';
        }

        try {
            $email = new \SendGrid\Mail\Mail();
            $email->setFrom($this->smtp['username'], $this->from_name);
            $email->setSubject($this->subject);
            $email->addTo($this->to);
            $email->addReplyTo($this->from_email, $this->from_name);

            // Prepare email body
            $body = "<html><body>";
            $body .= "<h2>New Contact Form Submission</h2>";
            $body .= "<table style='border-collapse: collapse; width: 100%;'>";
            
            foreach ($this->messages as $message) {
                if (!empty($message['label'])) {
                    $body .= "<tr>";
                    $body .= "<td style='padding: 8px; border: 1px solid #ddd;'><strong>{$message['label']}:</strong></td>";
                    $body .= "<td style='padding: 8px; border: 1px solid #ddd;'>{$message['message']}</td>";
                    $body .= "</tr>";
                }
            }
            
            $body .= "</table>";
            $body .= "</body></html>";

            $email->addContent("text/html", $body);

            $sendgrid = new \SendGrid($this->smtp['api_key']);
            $response = $sendgrid->send($email);

            if ($response->statusCode() == 202) {
                return 'OK';
            } else {
                return 'Failed to send email. Status code: ' . $response->statusCode();
            }
        } catch (Exception $e) {
            return "Message could not be sent. Error: " . $e->getMessage();
        }
    }
}
?> 