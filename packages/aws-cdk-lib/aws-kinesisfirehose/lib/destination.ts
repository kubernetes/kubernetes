import type { Construct, IDependable } from 'constructs';
import type { CfnDeliveryStream } from './kinesisfirehose.generated';

/**
 * An Amazon Data Firehose delivery stream destination configuration.
 */
export interface DestinationConfig {
  /**
   * S3 destination configuration properties.
   *
   * @default - S3 destination is not used.
   */
  readonly extendedS3DestinationConfiguration?: CfnDeliveryStream.ExtendedS3DestinationConfigurationProperty;
  /**
   * Datadog destination configuration properties.
   *
   * @default - HTTP destination is not used.
   */
  readonly httpEndpointDestinationConfiguration?: CfnDeliveryStream.HttpEndpointDestinationConfigurationProperty;

  /**
   * Any resources that were created by the destination when binding it to the stack that must be deployed before the delivery stream is deployed.
   *
   * @default []
   */
  readonly dependables?: IDependable[];
}

/**
 * Options when binding a destination to a delivery stream.
 */
export interface DestinationBindOptions {
}

/**
 * An Amazon Data Firehose delivery stream destination.
 */
export interface IDestination {
  /**
   * Binds this destination to the Amazon Data Firehose delivery stream.
   *
   * Implementers should use this method to bind resources to the stack and initialize values using the provided stream.
   */
  bind(scope: Construct, options: DestinationBindOptions): DestinationConfig;
}
