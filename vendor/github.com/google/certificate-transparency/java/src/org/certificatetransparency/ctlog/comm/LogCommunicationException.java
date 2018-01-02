package org.certificatetransparency.ctlog.comm;

import org.certificatetransparency.ctlog.CertificateTransparencyException;

/**
 * Indicates the log was unreadable  - HTTP communication with it was not possible.
 */
public class LogCommunicationException extends CertificateTransparencyException {
  private static final long serialVersionUID = 1L;

  public LogCommunicationException(String message) {
    super(message);
  }

  public LogCommunicationException(String message, Throwable cause) {
    super(message, cause);
  }
}
