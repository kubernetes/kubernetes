import { Token, UnscopedValidationError } from 'aws-cdk-lib';
import type { IRole } from 'aws-cdk-lib/aws-iam';
import type { CfnChannel } from 'aws-cdk-lib/aws-medialive';
import type { IBucket } from 'aws-cdk-lib/aws-s3';
import type { IStringParameter } from 'aws-cdk-lib/aws-ssm';
import { lit } from 'aws-cdk-lib/core/lib/helpers-internal';

/**
 * Options for a URL-based file location (`FileLocation.url`).
 */
export interface FileLocationOptions {
  /**
   * The username for accessing the upstream system.
   * @default - no credentials
   */
  readonly username?: string;
  /**
   * The SSM parameter that holds the password for accessing the upstream system. The channel
   * role is granted read access to the parameter automatically.
   * @default - no credentials
   */
  readonly password?: IStringParameter;
}

/**
 * A reference to a file MediaLive reads at runtime — for example an input-loss slate image,
 * an avail-blanking image, a burn-in caption font, or a color-correction LUT.
 *
 * Use the static factory methods to create one from an S3 bucket (which auto-grants the
 * channel role read access) or from a raw URL.
 */
export abstract class FileLocation {
  /**
   * Reference a file in an S3 bucket. Automatically grants the channel role read access.
   *
   * @param bucket The S3 bucket containing the file.
   * @param key The object key within the bucket (e.g. 'slates/offline.png').
   */
  public static fromBucket(bucket: IBucket, key: string): FileLocation {
    return new S3FileLocation(bucket, key);
  }

  /**
   * Reference a file by URL (e.g. an `https://` endpoint or an `s3ssl://` path).
   *
   * @param url The URL MediaLive reads the file from.
   * @param options Optional credentials.
   */
  public static url(url: string, options?: FileLocationOptions): FileLocation {
    return new UrlFileLocation(url, options);
  }

  /** @internal */
  public abstract _bind(): CfnChannel.InputLocationProperty;

  /** @internal */
  public _grantRead(_role: IRole): void {}
}

/** @internal */
class S3FileLocation extends FileLocation {
  private readonly url: string;
  constructor(private readonly bucket: IBucket, key: string) {
    super();
    this.url = `s3ssl://${bucket.bucketName}/${key}`;
  }
  public _bind(): CfnChannel.InputLocationProperty {
    return { uri: this.url };
  }
  public override _grantRead(role: IRole): void {
    this.bucket.grantRead(role);
  }
}

/** @internal */
class UrlFileLocation extends FileLocation {
  constructor(private readonly url: string, private readonly options?: FileLocationOptions) { super(); }
  public _bind(): CfnChannel.InputLocationProperty {
    return {
      uri: this.url,
      username: this.options?.username,
      // MediaLive's InputLocation passwordParam expects the SSM parameter reference in
      // `ssm://<name>` form, not the bare parameter name.
      passwordParam: this.options?.password ? `ssm://${this.options.password.parameterName}` : undefined,
    };
  }
  public override _grantRead(role: IRole): void {
    // MediaLive reads the password from SSM Parameter Store at channel runtime, so the
    // channel role needs read access to the parameter (scoped to the parameter ARN).
    this.options?.password?.grantRead(role);
  }
}

/**
 * The S3 location of a 3D LUT (look-up table) file used by a color-correction rule. MediaLive
 * reads the LUT from S3 at runtime, so the file must be in an S3 bucket — the URI must use the
 * `s3://` or `s3ssl://` protocol. Unlike a `FileLocation`, a LUT has no credentials.
 *
 * Use the static factory methods to create one from an S3 bucket (which uses the secure
 * `s3ssl://` form and auto-grants the channel role read access) or a raw S3 URL.
 */
export abstract class Lut {
  /**
   * Reference a LUT file in an S3 bucket. Uses the secure `s3ssl://` form and automatically
   * grants the channel role read access.
   *
   * @param bucket The S3 bucket containing the LUT file.
   * @param key The object key within the bucket (e.g. 'luts/rec709.cube').
   */
  public static fromBucket(bucket: IBucket, key: string): Lut {
    return new S3Lut(bucket, key);
  }

  /**
   * Reference a LUT file by S3 URL. The URL must use the `s3://` or `s3ssl://` protocol.
   *
   * @param url The `s3://` or `s3ssl://` URL of the LUT file.
   */
  public static url(url: string): Lut {
    if (!Token.isUnresolved(url) && !url.startsWith('s3://') && !url.startsWith('s3ssl://')) {
      throw new UnscopedValidationError(
        lit`LutProtocol`,
        `LUT location must be an s3:// or s3ssl:// URL, got ${JSON.stringify(url)}`,
      );
    }
    return new UrlLut(url);
  }

  /** @internal */
  public abstract _bind(): string;

  /** @internal */
  public _grantRead(_role: IRole): void {}
}

/** @internal */
class S3Lut extends Lut {
  private readonly url: string;
  constructor(private readonly bucket: IBucket, key: string) {
    super();
    this.url = `s3ssl://${bucket.bucketName}/${key}`;
  }
  public _bind(): string {
    return this.url;
  }
  public override _grantRead(role: IRole): void {
    this.bucket.grantRead(role);
  }
}

/** @internal */
class UrlLut extends Lut {
  constructor(private readonly url: string) { super(); }
  public _bind(): string {
    return this.url;
  }
}
