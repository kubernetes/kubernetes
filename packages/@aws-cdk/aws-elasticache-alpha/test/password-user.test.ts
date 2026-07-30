import { Stack, SecretValue } from 'aws-cdk-lib';
import { Template, Match } from 'aws-cdk-lib/assertions';
import { PasswordUser, AccessControl, UserEngine } from '../lib';

describe('PasswordUser', () => {
  describe('validation errors', () => {
    let stack: Stack;
    beforeEach(() => {
      stack = new Stack();
    });

    test.each([
      {
        testDescription: 'when no passwords provided throws validation error',
        passwords: [],
        errorMessage: 'Password authentication requires 1-2 passwords.',
      },
      {
        testDescription: 'when more than 2 passwords provided throws validation error',
        passwords: [
          SecretValue.secretsManager('secretvalue-1234'),
          SecretValue.secretsManager('secretvalue-12345'),
          SecretValue.secretsManager('secretvalue-123456'),
        ],
        errorMessage: 'Password authentication requires 1-2 passwords.',
      },
    ])('$testDescription', ({ passwords, errorMessage }) => {
      expect(() => new PasswordUser(stack, 'TestUser', {
        userId: 'test-user',
        accessControl: AccessControl.fromAccessString('on ~* +@all'),
        passwords,
      })).toThrow(errorMessage);
    });
  });

  describe('constructor', () => {
    let stack: Stack;
    beforeEach(() => {
      stack = new Stack();
    });

    test('creates user with minimal required properties (single password)', () => {
      new PasswordUser(stack, 'TestUser', {
        userId: 'test-user',
        accessControl: AccessControl.fromAccessString('on ~* +@all'),
        passwords: [SecretValue.unsafePlainText('secretvalue-1234')],
      });

      const template = Template.fromStack(stack);
      template.hasResourceProperties('AWS::ElastiCache::User', {
        Engine: 'valkey',
        UserId: 'test-user',
        UserName: 'test-user',
        AccessString: 'on ~* +@all',
        AuthenticationMode: {
          Type: 'password',
          Passwords: ['secretvalue-1234'],
        },
        NoPasswordRequired: false,
        Passwords: Match.absent(),
      });
    });

    test('creates user with two passwords', () => {
      new PasswordUser(stack, 'TestUser', {
        userId: 'test-user',
        accessControl: AccessControl.fromAccessString('on ~* +@all'),
        passwords: [
          SecretValue.unsafePlainText('secretvalue-1234'),
          SecretValue.unsafePlainText('secretvalue-12345'),
        ],
      });

      const template = Template.fromStack(stack);
      template.hasResourceProperties('AWS::ElastiCache::User', {
        AuthenticationMode: {
          Type: 'password',
          Passwords: ['secretvalue-1234', 'secretvalue-12345'],
        },
      });
    });

    test('creates user with all possible properties', () => {
      new PasswordUser(stack, 'TestUser', {
        userId: 'test-user',
        accessControl: AccessControl.fromAccessString('on ~app:* +@read +@write'),
        engine: UserEngine.REDIS,
        userName: 'test-user-name',
        passwords: [SecretValue.secretsManager('my-secret')],
      });

      const template = Template.fromStack(stack);
      template.hasResourceProperties('AWS::ElastiCache::User', {
        Engine: 'redis',
        UserId: 'test-user',
        UserName: 'test-user-name',
        AccessString: 'on ~app:* +@read +@write',
        AuthenticationMode: {
          Type: 'password',
          Passwords: Match.anyValue(),
        },
        NoPasswordRequired: false,
        Passwords: Match.absent(),
      });
    });

    test('creates exactly one ElastiCache user resource', () => {
      new PasswordUser(stack, 'TestUser', {
        userId: 'test-user',
        accessControl: AccessControl.fromAccessString('on ~* +@all'),
        passwords: [SecretValue.secretsManager('my-secret')],
      });

      const template = Template.fromStack(stack);
      template.resourceCountIs('AWS::ElastiCache::User', 1);
    });
  });

  describe('properties', () => {
    let stack: Stack;
    beforeEach(() => {
      stack = new Stack();
    });

    test('exposes correct properties', () => {
      const user = new PasswordUser(stack, 'TestUser', {
        userId: 'test-user-id',
        userName: 'test-user-name',
        engine: UserEngine.VALKEY,
        accessControl: AccessControl.fromAccessString('on ~app:* +@read'),
        passwords: [SecretValue.secretsManager('secretvalue-1234')],
      });

      expect(user.userId).toBe('test-user-id');
      expect(user.userName).toBe('test-user-name');
      expect(user.engine?.engineType).toBe('valkey');
      expect(user.accessString).toBe('on ~app:* +@read');
      expect(user.userArn).toBeDefined();
      expect(user.userStatus).toBeDefined();
    });

    test('userName defaults to userId when not provided', () => {
      const user = new PasswordUser(stack, 'TestUser', {
        userId: 'my-user-id',
        engine: UserEngine.REDIS,
        accessControl: AccessControl.fromAccessString('on ~* +@all'),
        passwords: [SecretValue.secretsManager('my-secret')],
      });

      expect(user.userName).toBe('my-user-id');
      expect(user.engine?.engineType).toBe('redis');
    });

    test('handles mixed password types', () => {
      new PasswordUser(stack, 'TestUser', {
        userId: 'test-user',
        accessControl: AccessControl.fromAccessString('on ~* +@all'),
        passwords: [
          SecretValue.secretsManager('elasticache/user/password'),
          SecretValue.unsafePlainText('plaintext-password'),
        ],
      });

      const template = Template.fromStack(stack);
      template.hasResourceProperties('AWS::ElastiCache::User', {
        AuthenticationMode: {
          Type: 'password',
          Passwords: [Match.anyValue(), 'plaintext-password'],
        },
      });
    });
  });

  describe('isPasswordUser', () => {
    test('returns true for PasswordUser instances', () => {
      const stack = new Stack();
      const user = new PasswordUser(stack, 'TestUser', {
        userId: 'test-user',
        engine: UserEngine.VALKEY,
        accessControl: AccessControl.fromAccessString('on ~* +@all'),
        passwords: [SecretValue.secretsManager('my-secret')],
      });

      expect(PasswordUser.isPasswordUser(user)).toBe(true);
    });

    test('returns false for non-PasswordUser objects', () => {
      expect(PasswordUser.isPasswordUser({})).toBe(false);
      expect(PasswordUser.isPasswordUser(null)).toBe(false);
      expect(PasswordUser.isPasswordUser(undefined)).toBe(false);
      expect(PasswordUser.isPasswordUser('string')).toBe(false);
      expect(PasswordUser.isPasswordUser(123)).toBe(false);
    });

    test('returns false for imported users (not actual PasswordUser instances)', () => {
      const stack = new Stack();
      const importedUser = PasswordUser.fromUserId(stack, 'ImportedUser', 'test-user');

      expect(PasswordUser.isPasswordUser(importedUser)).toBe(false);
    });
  });

  describe('import methods', () => {
    let stack: Stack;
    beforeEach(() => {
      stack = new Stack();
    });

    test('fromUserAttributes works with valid userArn', () => {
      const user = PasswordUser.fromUserAttributes(stack, 'ImportedUser', {
        userArn: 'arn:aws:elasticache:us-east-1:123456789012:user:my-user',
      });

      expect(user.userId).toBe('my-user');
      expect(user.userArn).toBe('arn:aws:elasticache:us-east-1:123456789012:user:my-user');
      expect(user.userName).toBe(undefined);
      expect(user.engine).toBe(undefined);
    });

    test('fromUserAttributes works with userId only', () => {
      const user = PasswordUser.fromUserAttributes(stack, 'ImportedUser', {
        userId: 'imported-user',
      });

      expect(user.userId).toBe('imported-user');
      expect(user.userArn).toContain('imported-user');
      expect(user.userName).toBe(undefined);
      expect(user.engine).toBe(undefined);
    });

    test('fromUserAttributes preserves engine when provided', () => {
      const user = PasswordUser.fromUserAttributes(stack, 'ImportedUser', {
        userId: 'test-user',
        engine: UserEngine.REDIS,
      });

      expect(user.engine).toBe(UserEngine.REDIS);
    });

    test('fromUserAttributes preserves userName when provided', () => {
      const user = PasswordUser.fromUserAttributes(stack, 'ImportedUser', {
        userId: 'test-user',
        userName: 'custom-name',
      });

      expect(user.userName).toBe('custom-name');
    });

    test('fromUserAttributes works with both engine and userName', () => {
      const user = PasswordUser.fromUserAttributes(stack, 'ImportedUser', {
        userId: 'test-user',
        engine: UserEngine.REDIS,
        userName: 'custom-name',
      });

      expect(user.userId).toBe('test-user');
      expect(user.engine).toBe(UserEngine.REDIS);
      expect(user.userName).toBe('custom-name');
    });

    test('fromUserAttributes with userArn preserves additional attributes', () => {
      const arn = 'arn:aws:elasticache:us-east-1:123456789012:user:my-user';
      const user = PasswordUser.fromUserAttributes(stack, 'ImportedUser', {
        userArn: arn,
        engine: UserEngine.VALKEY,
        userName: 'display-name',
      });

      expect(user.userArn).toBe(arn);
      expect(user.userId).toBe('my-user');
      expect(user.engine?.engineType).toBe('valkey');
      expect(user.userName).toBe('display-name');
    });

    test('fromUserId creates user with correct properties', () => {
      const user = PasswordUser.fromUserId(stack, 'ImportedUser', 'my-user-id');

      expect(user.userId).toBe('my-user-id');
      expect(user.userArn).toContain('my-user-id');
      expect(user.userName).toBe(undefined);
      expect(user.engine).toBe(undefined);
    });

    test('fromUserArn creates user with correct properties', () => {
      const arn = 'arn:aws:elasticache:us-west-2:123456789012:user:test-user';
      const user = PasswordUser.fromUserArn(stack, 'ImportedUser', arn);

      expect(user.userId).toBe('test-user');
      expect(user.userArn).toBe(arn);
      expect(user.userName).toBe(undefined);
      expect(user.engine).toBe(undefined);
    });

    test.each([
      {
        testDescription: 'when passing both userId and userArn throws validation error',
        userArn: 'arn:aws:elasticache:us-east-1:999999999999:user:test-user',
        userId: 'test-user',
        errorMessage: 'Only one of userArn or userId can be provided.',
      },
      {
        testDescription: 'when passing neither userId nor userArn throws validation error',
        errorMessage: 'One of userId or userArn is required.',
      },
      {
        testDescription: 'when passing invalid userArn (no user id) throws validation error',
        userArn: 'arn:aws:elasticache:us-east-1:999999999999:user',
        errorMessage: 'Unable to extract user id from ARN.',
      },
    ])('$testDescription', ({ userArn, userId, errorMessage }) => {
      expect(() => PasswordUser.fromUserAttributes(stack, 'ImportedUser', { userArn, userId })).toThrow(errorMessage);
    });
  });
});

