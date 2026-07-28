import * as integRunner from '../lib/integration-test-runner';
import { main } from '../lib/main';
import * as preflight from '../lib/preflight';

describe('main function', () => {
  let consoleLogSpy: jest.SpyInstance;
  let deployIntegTestsSpy: jest.SpyInstance;
  let shouldRunIntegTestsSpy: jest.SpyInstance;

  beforeEach(() => {
    jest.clearAllMocks();
    consoleLogSpy = jest.spyOn(console, 'log').mockImplementation();
    deployIntegTestsSpy = jest.spyOn(integRunner, 'deployIntegTests').mockResolvedValue(undefined);
    shouldRunIntegTestsSpy = jest.spyOn(preflight, 'shouldRunIntegTests');
  });

  afterEach(() => {
    consoleLogSpy.mockRestore();
    deployIntegTestsSpy.mockRestore();
    shouldRunIntegTestsSpy.mockRestore();
  });

  const baseConfig = {
    endpoint: 'https://test-endpoint.com',
    pool: 'test-pool',
    atmosphereRoleArn: 'arn:aws:iam::123456789:role/test',
  };

  describe('preflight check behavior', () => {
    it('should skip preflight when GitHub context is not available', async () => {
      await main(baseConfig);

      expect(shouldRunIntegTestsSpy).not.toHaveBeenCalled();
      expect(consoleLogSpy).toHaveBeenCalledWith(
        'GitHub context not available, skipping preflight check (manual run or workflow_dispatch)',
      );
      expect(deployIntegTestsSpy).toHaveBeenCalled();
    });

    it('should skip preflight when only GITHUB_TOKEN is available', async () => {
      await main({
        ...baseConfig,
        githubToken: 'test-token',
      });

      expect(shouldRunIntegTestsSpy).not.toHaveBeenCalled();
      expect(deployIntegTestsSpy).toHaveBeenCalled();
    });

    it('should skip preflight when GITHUB_REPOSITORY is missing', async () => {
      await main({
        ...baseConfig,
        githubToken: 'test-token',
        prNumber: '123',
      });

      expect(shouldRunIntegTestsSpy).not.toHaveBeenCalled();
      expect(deployIntegTestsSpy).toHaveBeenCalled();
    });

    it('should skip preflight when PR_NUMBER is missing', async () => {
      await main({
        ...baseConfig,
        githubToken: 'test-token',
        githubRepository: 'aws/aws-cdk',
      });

      expect(shouldRunIntegTestsSpy).not.toHaveBeenCalled();
      expect(deployIntegTestsSpy).toHaveBeenCalled();
    });

    it('should run preflight when all GitHub context is available', async () => {
      shouldRunIntegTestsSpy.mockResolvedValue({
        shouldRun: true,
        reason: 'PR has snapshot changes and is approved by CDK team member',
      });

      await main({
        ...baseConfig,
        githubToken: 'test-token',
        githubRepository: 'aws/aws-cdk',
        prNumber: '123',
      });

      expect(shouldRunIntegTestsSpy).toHaveBeenCalledWith({
        githubToken: 'test-token',
        owner: 'aws',
        repo: 'aws-cdk',
        prNumber: 123,
      });
      expect(deployIntegTestsSpy).toHaveBeenCalled();
    });

    it('should skip deployment when preflight returns shouldRun=false', async () => {
      shouldRunIntegTestsSpy.mockResolvedValue({
        shouldRun: false,
        reason: 'No snapshot changes detected in this PR',
      });

      await main({
        ...baseConfig,
        githubToken: 'test-token',
        githubRepository: 'aws/aws-cdk',
        prNumber: '456',
      });

      expect(shouldRunIntegTestsSpy).toHaveBeenCalled();
      expect(consoleLogSpy).toHaveBeenCalledWith('Skipping integration test deployment.');
      expect(deployIntegTestsSpy).not.toHaveBeenCalled();
    });

    it('should proceed with deployment when preflight returns shouldRun=true', async () => {
      shouldRunIntegTestsSpy.mockResolvedValue({
        shouldRun: true,
        reason: 'PR has snapshot changes and is approved by CDK team member',
      });

      await main({
        ...baseConfig,
        githubToken: 'test-token',
        githubRepository: 'aws/aws-cdk',
        prNumber: '789',
      });

      expect(deployIntegTestsSpy).toHaveBeenCalledWith({
        atmosphereRoleArn: 'arn:aws:iam::123456789:role/test',
        endpoint: 'https://test-endpoint.com',
        pool: 'test-pool',
      });
    });

    it('should log preflight result reason', async () => {
      shouldRunIntegTestsSpy.mockResolvedValue({
        shouldRun: true,
        reason: 'PR has snapshot changes and is approved by CDK team member @maintainer',
      });

      await main({
        ...baseConfig,
        githubToken: 'test-token',
        githubRepository: 'aws/aws-cdk',
        prNumber: '123',
      });

      expect(consoleLogSpy).toHaveBeenCalledWith('Running preflight check for PR #123...');
      expect(consoleLogSpy).toHaveBeenCalledWith(
        'Preflight result: PR has snapshot changes and is approved by CDK team member @maintainer',
      );
    });
  });

  describe('environment variable validation', () => {
    it('should throw when endpoint is missing', async () => {
      await expect(main({
        pool: 'test-pool',
        atmosphereRoleArn: 'arn:aws:iam::123456789:role/test',
      })).rejects.toThrow('CDK_ATMOSPHERE_ENDPOINT environment variable is required');
    });

    it('should throw when pool is missing', async () => {
      await expect(main({
        endpoint: 'https://test-endpoint.com',
        atmosphereRoleArn: 'arn:aws:iam::123456789:role/test',
      })).rejects.toThrow('CDK_ATMOSPHERE_POOL environment variable is required');
    });

    it('should throw when atmosphereRoleArn is missing', async () => {
      await expect(main({
        endpoint: 'https://test-endpoint.com',
        pool: 'test-pool',
      })).rejects.toThrow('CDK_ATMOSPHERE_OIDC_ROLE environment variable is required');
    });

    it('should not validate env vars if preflight skips deployment', async () => {
      shouldRunIntegTestsSpy.mockResolvedValue({
        shouldRun: false,
        reason: 'No snapshot changes',
      });

      // Missing required env vars but should not throw because preflight skips
      await main({
        githubToken: 'test-token',
        githubRepository: 'aws/aws-cdk',
        prNumber: '123',
      });

      expect(deployIntegTestsSpy).not.toHaveBeenCalled();
    });
  });

  describe('input validation', () => {
    it('should throw on invalid GITHUB_REPOSITORY format (no slash)', async () => {
      await expect(main({
        ...baseConfig,
        githubToken: 'test-token',
        githubRepository: 'invalid-repo-format',
        prNumber: '123',
      })).rejects.toThrow('Invalid GITHUB_REPOSITORY format: "invalid-repo-format". Expected format: owner/repo');
    });

    it('should throw on invalid GITHUB_REPOSITORY format (too many slashes)', async () => {
      await expect(main({
        ...baseConfig,
        githubToken: 'test-token',
        githubRepository: 'aws/aws-cdk/extra',
        prNumber: '123',
      })).rejects.toThrow('Invalid GITHUB_REPOSITORY format: "aws/aws-cdk/extra". Expected format: owner/repo');
    });

    it('should throw on invalid GITHUB_REPOSITORY format (empty owner)', async () => {
      await expect(main({
        ...baseConfig,
        githubToken: 'test-token',
        githubRepository: '/aws-cdk',
        prNumber: '123',
      })).rejects.toThrow('Invalid GITHUB_REPOSITORY format: "/aws-cdk". Expected format: owner/repo');
    });

    it('should throw on invalid GITHUB_REPOSITORY format (empty repo)', async () => {
      await expect(main({
        ...baseConfig,
        githubToken: 'test-token',
        githubRepository: 'aws/',
        prNumber: '123',
      })).rejects.toThrow('Invalid GITHUB_REPOSITORY format: "aws/". Expected format: owner/repo');
    });

    it('should throw on invalid PR number (non-numeric)', async () => {
      await expect(main({
        ...baseConfig,
        githubToken: 'test-token',
        githubRepository: 'aws/aws-cdk',
        prNumber: 'not-a-number',
      })).rejects.toThrow('Invalid PR number: "not-a-number". Expected a positive integer.');
    });

    it('should throw on invalid PR number (zero)', async () => {
      await expect(main({
        ...baseConfig,
        githubToken: 'test-token',
        githubRepository: 'aws/aws-cdk',
        prNumber: '0',
      })).rejects.toThrow('Invalid PR number: "0". Expected a positive integer.');
    });

    it('should throw on invalid PR number (negative)', async () => {
      await expect(main({
        ...baseConfig,
        githubToken: 'test-token',
        githubRepository: 'aws/aws-cdk',
        prNumber: '-5',
      })).rejects.toThrow('Invalid PR number: "-5". Expected a positive integer.');
    });

    it('should accept valid repository and PR number', async () => {
      shouldRunIntegTestsSpy.mockResolvedValue({
        shouldRun: true,
        reason: 'Test',
      });

      await main({
        ...baseConfig,
        githubToken: 'test-token',
        githubRepository: 'aws/aws-cdk',
        prNumber: '12345',
      });

      expect(shouldRunIntegTestsSpy).toHaveBeenCalledWith({
        githubToken: 'test-token',
        owner: 'aws',
        repo: 'aws-cdk',
        prNumber: 12345,
      });
    });
  });
});
