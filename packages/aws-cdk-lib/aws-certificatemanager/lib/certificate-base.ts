import type { ICertificate } from './certificate';
import * as cloudwatch from '../../aws-cloudwatch';
import { Stats } from '../../aws-cloudwatch';
import { Duration, Resource } from '../../core';
import type { CertificateReference } from '../../interfaces/generated/aws-certificatemanager-interfaces.generated';

/**
 * Shared implementation details of ICertificate implementations.
 *
 * @internal
 */
export abstract class CertificateBase extends Resource implements ICertificate {
  public abstract readonly certificateArn: string;

  /**
   * If the certificate is provisionned in a different region than the
   * containing stack, this should be the region in which the certificate lives
   * so we can correctly create `Metric` instances.
   */
  protected readonly region?: string;

  public get certificateRef(): CertificateReference {
    return {
      certificateArn: this.certificateArn,
    };
  }

  public metricDaysToExpiry(props?: cloudwatch.MetricOptions): cloudwatch.Metric {
    return new cloudwatch.Metric({
      period: Duration.days(1),
      ...props,
      dimensionsMap: { CertificateArn: this.certificateArn },
      metricName: 'DaysToExpiry',
      namespace: 'AWS/CertificateManager',
      region: this.region,
      statistic: Stats.MINIMUM,
    });
  }
}
