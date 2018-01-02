package org.certificatetransparency.ctlog.serialization;

import org.certificatetransparency.ctlog.CertificateTransparencyException;

/**
 * Error serializing / deserializing data.
 */
public class SerializationException extends CertificateTransparencyException {
  private static final long serialVersionUID = 1L;

  public SerializationException(String message) {
    super(message);
  }

  public SerializationException(String message, Throwable cause) {
    super(message, cause);
  }
}
