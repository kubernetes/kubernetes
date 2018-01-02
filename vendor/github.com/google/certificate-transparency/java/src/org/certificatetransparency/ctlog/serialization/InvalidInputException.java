package org.certificatetransparency.ctlog.serialization;

import org.certificatetransparency.ctlog.CertificateTransparencyException;

/**
 * Input certificates or log key are invalid.
 */
public class InvalidInputException extends CertificateTransparencyException {
  private static final long serialVersionUID = 1L;

  public InvalidInputException(String message, Throwable cause) {
    super(message, cause);
  }
}
