import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from config import settings


def is_email_delivery_configured() -> bool:
    required = [
        settings.get_string("SMTP_HOST", ""),
        str(settings.get_int("SMTP_PORT", 0)),
        settings.get_string("SMTP_USER", ""),
        settings.get_string("SMTP_PASSWORD", ""),
        settings.get_string("SMTP_FROM_EMAIL", ""),
    ]
    return all(str(item or "").strip() for item in required)


def _format_password_reset_ttl(ttl_seconds: int) -> str:
    total_seconds = max(1, int(ttl_seconds or 0))
    if total_seconds < 60:
        unit = "second" if total_seconds == 1 else "seconds"
        return f"{total_seconds} {unit}"
    total_minutes = total_seconds // 60
    unit = "minute" if total_minutes == 1 else "minutes"
    return f"{total_minutes} {unit}"


def send_password_reset_otp_email(recipient_email: str, otp_code: str, ttl_seconds: int) -> None:
    if not is_email_delivery_configured():
        raise RuntimeError("SMTP email delivery is not configured")

    smtp_host = settings.get_string("SMTP_HOST", "")
    smtp_port = settings.get_int("SMTP_PORT", 587)
    smtp_user = settings.get_string("SMTP_USER", "")
    smtp_password = settings.get_string("SMTP_PASSWORD", "")
    from_email = settings.get_string("SMTP_FROM_EMAIL", smtp_user) or smtp_user
    from_name = settings.get_string("SMTP_FROM_NAME", "Document Management System")
    ttl_label = _format_password_reset_ttl(ttl_seconds)

    subject = "Your password reset code"
    text_body = (
        f"Your password reset verification code is {otp_code}. "
        f"It will expire in {ttl_label}. "
        "If you did not request this change, you can ignore this email."
    )
    html_body = f"""
    <html>
      <body style="margin:0;padding:24px;background:#fff8f3;font-family:Segoe UI,Tahoma,sans-serif;color:#1f1b1a;">
        <div style="max-width:560px;margin:0 auto;background:#ffffff;border:1px solid #f4d5c7;border-radius:20px;padding:28px 28px 24px;">
          <div style="font-size:12px;letter-spacing:0.14em;text-transform:uppercase;color:#b45925;font-weight:700;margin-bottom:12px;">
            Password Reset
          </div>
          <h2 style="margin:0 0 12px;font-size:28px;line-height:1.15;">Use this verification code</h2>
          <p style="margin:0 0 18px;font-size:15px;line-height:1.7;color:#6f5f58;">
            We received a request to reset your Document Management password.
            Enter the code below to continue.
          </p>
          <div style="margin:0 0 18px;padding:16px 18px;border-radius:18px;background:linear-gradient(120deg,#ffefe2,#fff8f3);border:1px solid #ffd6bf;text-align:center;">
            <div style="font-size:32px;letter-spacing:0.32em;font-weight:800;color:#ef3e23;">{otp_code}</div>
          </div>
          <p style="margin:0 0 10px;font-size:14px;line-height:1.7;color:#6f5f58;">
            This code expires in <strong>{ttl_label}</strong>.
          </p>
          <p style="margin:0;font-size:13px;line-height:1.7;color:#8c7a72;">
            If you did not request a password reset, you can safely ignore this email.
          </p>
        </div>
      </body>
    </html>
    """.strip()

    message = MIMEMultipart("alternative")
    message["Subject"] = subject
    message["From"] = f"{from_name} <{from_email}>"
    message["To"] = recipient_email
    message.attach(MIMEText(text_body, "plain", "utf-8"))
    message.attach(MIMEText(html_body, "html", "utf-8"))

    with smtplib.SMTP(smtp_host, smtp_port, timeout=20) as smtp:
        smtp.ehlo()
        smtp.starttls()
        smtp.ehlo()
        smtp.login(smtp_user, smtp_password)
        smtp.sendmail(from_email, [recipient_email], message.as_string())
