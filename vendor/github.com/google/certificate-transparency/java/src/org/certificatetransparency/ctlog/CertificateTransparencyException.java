package org.certificatetransparency.ctlog;

/**
 * Base class for CT errors.
 */
public class CertificateTransparencyException extends RuntimeException {
  private static final long serialVersionUID = 1L;

  public CertificateTransparencyException(String message) {
    super(message);
  }

  public CertificateTransparencyException(String message, Throwable cause) {
    super(message + ": " + cause.getMessage(), cause);
  }
}
