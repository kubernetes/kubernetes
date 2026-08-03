import type { ICertificateRef } from '../../../interfaces/generated/aws-certificatemanager-interfaces.generated';

/**
 * A certificate source for an ELBv2 listener
 */
export interface IListenerCertificate {
  /**
   * The ARN of the certificate to use
   */
  readonly certificateArn: string;
}

/**
 * A certificate source for an ELBv2 listener
 */
export class ListenerCertificate implements IListenerCertificate {
  /**
   * Use an ACM certificate as a listener certificate
   */
  public static fromCertificateManager(this: void, acmCertificate: ICertificateRef) {
    return new ListenerCertificate(acmCertificate.certificateRef.certificateArn);
  }

  /**
   * Use any certificate, identified by its ARN, as a listener certificate
   */
  public static fromArn(this: void, certificateArn: string) {
    return new ListenerCertificate(certificateArn);
  }

  /**
   * The ARN of the certificate to use
   */
  public readonly certificateArn: string;

  protected constructor(certificateArn: string) {
    this.certificateArn = certificateArn;
  }
}
