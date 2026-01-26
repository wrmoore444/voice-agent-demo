import os
import aiosmtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from loguru import logger
from typing import Dict, List
from datetime import datetime


async def send_call_summary_email(
    recipient_email: str,
    transcription: List[Dict],
    datapoints: Dict,
    agent_name: str = "AI Agent",
    conversation_uuid: str = None
):
    """
    Send email with call transcription and datapoints using Gmail SMTP
    
    Args:
        recipient_email: Email address to send to
        transcription: List of transcription messages
        datapoints: Extracted datapoints dictionary
        agent_name: Name of the agent
        conversation_id: Conversation UUID
    """
    try:
        # Get Gmail credentials from environment
        gmail_address = os.getenv("GMAIL_ADDRESS")
        gmail_password = os.getenv("GMAIL_APP_PASSWORD")
        
        if not gmail_address or not gmail_password:
            logger.error("Gmail credentials not found in environment variables")
            return False
        
        # Format transcription
        transcription_text = "\n".join([
            f"[{msg.get('timestamp', 'N/A')}] {msg.get('speaker', 'Unknown')}: {msg.get('text', '')}"
            for msg in transcription
        ])
        
        # Format datapoints
        datapoints_text = "\n".join([
            f"• {key.replace('_', ' ').title()}: {value}"
            for key, value in datapoints.items()
            if value and key != 'raw_transcript'
        ])
        
        # Create HTML email content
        html_content = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; max-width: 800px; margin: 0 auto; }}
                .header {{ background-color: #4CAF50; color: white; padding: 20px; text-align: center; border-radius: 5px; }}
                .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; background-color: #ffffff; }}
                .section h2 {{ color: #4CAF50; border-bottom: 2px solid #4CAF50; padding-bottom: 10px; margin-top: 0; }}
                .transcription {{ background-color: #f9f9f9; padding: 15px; border-left: 4px solid #4CAF50; white-space: pre-wrap; font-family: monospace; font-size: 12px; max-height: 500px; overflow-y: auto; }}
                .datapoints {{ background-color: #fff3cd; padding: 15px; border-left: 4px solid #ffc107; }}
                .datapoints pre {{ margin: 0; white-space: pre-wrap; font-family: Arial, sans-serif; }}
                .footer {{ text-align: center; margin-top: 30px; padding: 20px; color: #666; font-size: 12px; border-top: 1px solid #ddd; }}
                .timestamp {{ color: #666; font-size: 11px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>📞 Call Summary Report</h1>
                <p style="margin: 5px 0;">Agent: <strong>{agent_name}</strong></p>
                {f'<p style="margin: 5px 0;">Conversation UUID: {conversation_uuid}</p>' if conversation_uuid else ''}
                <p class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="section">
                <h2>📊 Extracted Datapoints</h2>
                <div class="datapoints">
                    <pre>{datapoints_text if datapoints_text else 'No datapoints extracted'}</pre>
                </div>
            </div>
            
            <div class="section">
                <h2>📝 Full Transcription</h2>
                <div class="transcription">{transcription_text if transcription_text else 'No transcription available'}</div>
            </div>
            
            <div class="footer">
                <p>This is an automated email from your AI Voice Agent system.</p>
            </div>
        </body>
        </html>
        """
        
        # Create plain text version
        plain_text = f"""
Call Summary Report
{'='*50}

Agent: {agent_name}
{'Conversation UUID: ' + conversation_uuid if conversation_uuid else ''}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

{'='*50}
EXTRACTED DATAPOINTS
{'='*50}
{datapoints_text if datapoints_text else 'No datapoints extracted'}

{'='*50}
FULL TRANSCRIPTION
{'='*50}
{transcription_text if transcription_text else 'No transcription available'}

---
This is an automated email from your AI Voice Agent system.
        """
        
        # Create message
        message = MIMEMultipart('alternative')
        message['Subject'] = f'Call Summary - {agent_name} - {datetime.now().strftime("%Y-%m-%d %H:%M")}'
        message['From'] = gmail_address
        message['To'] = recipient_email
        
        # Attach both plain text and HTML versions
        part1 = MIMEText(plain_text, 'plain')
        part2 = MIMEText(html_content, 'html')
        message.attach(part1)
        message.attach(part2)
        
        # Send email using Gmail SMTP with SSL (port 465)
        await aiosmtplib.send(
            message,
            hostname='smtp.gmail.com',
            port=465,
            use_tls=True,  # Changed from start_tls to use_tls for port 465
            username=gmail_address,
            password=gmail_password,
        )
        
        logger.info(f"Email sent successfully to {recipient_email} from {gmail_address}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to send email: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False