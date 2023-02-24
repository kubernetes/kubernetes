/*
Copyright 2023 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package remote

import (
	crand "crypto/rand"
	"crypto/rsa"
	"crypto/x509"
	"encoding/base64"
	"encoding/pem"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/session"
	"github.com/aws/aws-sdk-go/service/ec2"
	"github.com/aws/aws-sdk-go/service/ec2instanceconnect"
	"github.com/aws/aws-sdk-go/service/ssm"
	"github.com/google/uuid"
	"golang.org/x/crypto/ssh"
	"k8s.io/klog/v2"
	"sigs.k8s.io/yaml"
)

var _ Runner = (*AWSRunner)(nil)

var region = flag.String("region", "", "AWS region that the hosts live in (aws)")
var userDataFile = flag.String("user-data-file", "", "Path to user data to pass to created instances (aws)")
var instanceProfile = flag.String("instance-profile", "", "The name of the instance profile to assign to the node (aws)")
var instanceConnect = flag.Bool("ec2-instance-connect", true, "Use EC2 instance connect to generate a one time use key (aws)")

const defaultAWSInstanceType = "t3a.medium"
const amiIDTag = "Node-E2E-Test"

type AWSRunner struct {
	cfg               Config
	ec2Service        *ec2.EC2
	ec2icService      *ec2instanceconnect.EC2InstanceConnect
	ssmService        *ssm.SSM
	internalAWSImages []internalAWSImage
}

func NewAWSRunner(cfg Config) *AWSRunner {
	if cfg.InstanceNamePrefix == "" {
		cfg.InstanceNamePrefix = "tmp-node-e2e-" + uuid.New().String()[:8]
	}

	return &AWSRunner{cfg: cfg}
}

func (a *AWSRunner) Validate() error {
	if len(a.cfg.Hosts) == 0 && a.cfg.ImageConfigFile == "" && len(a.cfg.Images) == 0 {
		klog.Fatalf("Must specify one of --image-config-file, --hosts, --images.")
	}
	for _, img := range a.cfg.Images {
		if !strings.HasPrefix(img, "ami-") {
			return fmt.Errorf("invalid AMI id format for %q", img)
		}
	}
	sess, err := session.NewSession(&aws.Config{Region: region})
	if err != nil {
		klog.Fatalf("Unable to create AWS session, %s", err)
	}
	a.ec2Service = ec2.New(sess)
	a.ec2icService = ec2instanceconnect.New(sess)
	a.ssmService = ssm.New(sess)
	if a.internalAWSImages, err = a.prepareAWSImages(); err != nil {
		klog.Fatalf("While preparing AWS images: %v", err)
	}
	return nil
}

func (a *AWSRunner) StartTests(suite TestSuite, archivePath string, results chan *TestResult) (numTests int) {
	for i := range a.internalAWSImages {
		img := a.internalAWSImages[i]
		fmt.Printf("Initializing e2e tests using image %s /  %s.\n", img.imageDesc, img.amiID)
		numTests++
		go func() {
			results <- a.testAWSImage(suite, archivePath, img)
		}()
	}
	return
}

type AWSImageConfig struct {
	Images map[string]AWSImage `json:"images"`
}

type AWSImage struct {
	AmiID           string   `json:"ami_id"`
	SSMPath         string   `json:"ssm_path,omitempty"`
	InstanceType    string   `json:"instance_type,omitempty"`
	UserData        string   `json:"user_data_file,omitempty"`
	InstanceProfile string   `json:"instance_profile,omitempty"`
	Tests           []string `json:"tests,omitempty"`
}

type internalAWSImage struct {
	amiID string
	// The instance type (e.g. t3a.medium)
	instanceType string
	userData     []byte
	imageDesc    string
	// name of the instance profile
	instanceProfile string
}

func (a *AWSRunner) prepareAWSImages() ([]internalAWSImage, error) {
	var ret []internalAWSImage

	// Parse images from given config file and convert them to internalGCEImage.
	if a.cfg.ImageConfigFile != "" {
		configPath := a.cfg.ImageConfigFile
		if a.cfg.ImageConfigDir != "" {
			configPath = filepath.Join(a.cfg.ImageConfigDir, a.cfg.ImageConfigFile)
		}

		imageConfigData, err := os.ReadFile(configPath)
		if err != nil {
			return nil, fmt.Errorf("Could not read image config file provided: %w", err)
		}
		externalImageConfig := AWSImageConfig{Images: make(map[string]AWSImage)}
		err = yaml.Unmarshal(imageConfigData, &externalImageConfig)
		if err != nil {
			return nil, fmt.Errorf("could not parse image config file: %w", err)
		}

		for shortName, imageConfig := range externalImageConfig.Images {
			var amiID string
			if imageConfig.SSMPath != "" && imageConfig.AmiID == "" {
				amiID, err = a.getSSMImage(imageConfig.SSMPath)
				if err != nil {
					return nil, fmt.Errorf("could not retrieve a image based on SSM path %s, %w", imageConfig.SSMPath, err)
				}
			} else {
				amiID = imageConfig.AmiID
			}

			// user data can only be from an image config or the command line
			if *userDataFile != "" && imageConfig.UserData != "" {
				return nil, fmt.Errorf("can't specify userdata on both the command line and in an image config")
			}

			imageUserDataFile := *userDataFile
			if imageUserDataFile == "" && imageConfig.UserData != "" {
				imageUserDataFile = filepath.Join(a.cfg.ImageConfigDir, imageConfig.UserData)
			}
			var userdata []byte
			if imageUserDataFile != "" {
				userdata, err = os.ReadFile(imageUserDataFile)
				if err != nil {
					return nil, fmt.Errorf("reading userdata file %q, %w", imageUserDataFile, err)
				}
			}

			// the instance profile can from image config or the command line
			if *instanceProfile != "" && imageConfig.InstanceProfile != "" {
				return nil, fmt.Errorf("can't specify instance profile on both the command line and in an image config")
			}
			instanceProfile := *instanceProfile
			if instanceProfile == "" {
				instanceProfile = imageConfig.InstanceProfile
			}

			awsImage := internalAWSImage{
				amiID:           amiID,
				userData:        userdata,
				instanceType:    imageConfig.InstanceType,
				instanceProfile: instanceProfile,
				imageDesc:       shortName,
			}
			if awsImage.instanceType == "" {
				awsImage.instanceType = defaultAWSInstanceType
			}
			ret = append(ret, awsImage)
		}
	}

	if len(a.cfg.Images) > 0 {
		for _, img := range a.cfg.Images {
			ret = append(ret, internalAWSImage{
				amiID:        img,
				instanceType: defaultAWSInstanceType,
			})
		}
	}
	return ret, nil
}

func (a *AWSRunner) testAWSImage(suite TestSuite, archivePath string, imageConfig internalAWSImage) *TestResult {
	instance, err := a.getAWSInstance(imageConfig)
	if err != nil {
		return &TestResult{
			Err: fmt.Errorf("unable to create EC2 instance for image %s, %w", imageConfig.amiID, err),
		}
	}
	if a.cfg.DeleteInstances {
		defer a.deleteAWSInstance(instance.instanceID)
	}
	if instance.sshPublicKeyFile != "" && *instanceConnect {
		defer os.Remove(instance.sshPublicKeyFile)
	}
	deleteFiles := !a.cfg.DeleteInstances && a.cfg.Cleanup
	ginkgoFlagsStr := a.cfg.GinkgoFlags

	output, exitOk, err := RunRemote(RunRemoteConfig{
		suite:          suite,
		archive:        archivePath,
		host:           instance.instanceID,
		cleanup:        deleteFiles,
		imageDesc:      imageConfig.amiID,
		junitFileName:  instance.instanceID,
		testArgs:       a.cfg.TestArgs,
		ginkgoArgs:     ginkgoFlagsStr,
		systemSpecName: a.cfg.SystemSpecName,
		extraEnvs:      a.cfg.ExtraEnvs,
		runtimeConfig:  a.cfg.RuntimeConfig,
	})
	return &TestResult{
		Output: output,
		Err:    err,
		Host:   instance.instanceID,
		ExitOK: exitOk,
	}
}

func (a *AWSRunner) deleteAWSInstance(instanceID string) {
	klog.Infof("Terminating instance %q", instanceID)
	_, err := a.ec2Service.TerminateInstances(&ec2.TerminateInstancesInput{
		InstanceIds: []*string{&instanceID},
	})
	if err != nil {
		klog.Errorf("Error terminating instance %q: %v", instanceID, err)
	}
}

func (a *AWSRunner) getAWSInstance(img internalAWSImage) (*awsInstance, error) {
	// first see if we have an instance already running the desired image
	existing, err := a.ec2Service.DescribeInstances(&ec2.DescribeInstancesInput{
		Filters: []*ec2.Filter{
			{
				Name:   aws.String("instance-state-name"),
				Values: []*string{aws.String(ec2.InstanceStateNameRunning)},
			},
			{
				Name:   aws.String(fmt.Sprintf("tag:%s", amiIDTag)),
				Values: []*string{aws.String(img.amiID)},
			},
		},
	})
	if err != nil {
		return nil, err
	}

	var instance *ec2.Instance
	if len(existing.Reservations) > 0 && len(existing.Reservations[0].Instances) > 0 {
		instance = existing.Reservations[0].Instances[0]
		klog.Infof("reusing existing instance %s", *instance.InstanceId)
	} else {
		// no existing instance running that image, so we need to launch a new instance
		newInstance, err := a.launchNewInstance(img)
		if err != nil {
			return nil, err
		}
		instance = newInstance
		klog.Infof("launched new instance %s", *instance.InstanceId)
	}

	testInstance := &awsInstance{
		instanceID: *instance.InstanceId,
		instance:   instance,
	}

	instanceRunning := false
	createdSSHKey := false
	for i := 0; i < 30 && !instanceRunning; i++ {
		if i > 0 {
			time.Sleep(time.Second * 20)
		}

		var op *ec2.DescribeInstancesOutput
		op, err = a.ec2Service.DescribeInstances(&ec2.DescribeInstancesInput{
			InstanceIds: []*string{&testInstance.instanceID},
		})
		if err != nil {
			continue
		}
		instance := op.Reservations[0].Instances[0]
		if *instance.State.Name != ec2.InstanceStateNameRunning {
			continue
		}
		testInstance.publicIP = *instance.PublicIpAddress

		// generate a temporary SSH key and send it to the node via instance-connect
		if *instanceConnect && !createdSSHKey {
			err = a.assignNewSSHKey(testInstance)
			if err != nil {
				continue
			}
			createdSSHKey = true
		}

		klog.Infof("registering %s/%s", testInstance.instanceID, testInstance.publicIP)
		AddHostnameIP(testInstance.instanceID, testInstance.publicIP)

		// ensure that containerd or CRIO is running
		var output string
		output, err = SSH(testInstance.instanceID, "sh", "-c", "systemctl list-units  --type=service  --state=running | grep -e containerd -e crio")
		if err != nil {
			err = fmt.Errorf("instance %s not running containerd/crio daemon - Command failed: %s", testInstance.instanceID, output)
			continue
		}
		if !strings.Contains(output, "containerd.service") &&
			!strings.Contains(output, "crio.service") {
			err = fmt.Errorf("instance %s not running containerd/crio daemon: %s", testInstance.instanceID, output)
			continue
		}

		instanceRunning = true
	}

	if !instanceRunning {
		return nil, fmt.Errorf("instance %s is not running, %w", testInstance.instanceID, err)
	}
	return testInstance, nil
}

// assignNewSSHKey generates a new SSH key-pair and assigns it to the EC2 instance using EC2-instance connect. It then
// connects via SSH and makes the key permanent by writing it to ~/.ssh/authorized_keys
func (a *AWSRunner) assignNewSSHKey(testInstance *awsInstance) error {
	// create our new key
	key, err := generateSSHKeypair()
	if err != nil {
		return fmt.Errorf("creating SSH key, %w", err)
	}
	testInstance.sshKey = key
	_, err = a.ec2icService.SendSSHPublicKey(&ec2instanceconnect.SendSSHPublicKeyInput{
		InstanceId:     aws.String(testInstance.instanceID),
		InstanceOSUser: aws.String(GetSSHUser()),
		SSHPublicKey:   aws.String(string(key.public)),
	})
	if err != nil {
		return fmt.Errorf("sending SSH public key for serial console access, %w", err)
	}
	client, err := ssh.Dial("tcp", fmt.Sprintf("%s:22", testInstance.publicIP), &ssh.ClientConfig{
		User:            GetSSHUser(),
		HostKeyCallback: ssh.InsecureIgnoreHostKey(),
		Auth: []ssh.AuthMethod{
			ssh.PublicKeys(key.signer),
		},
	})
	if err != nil {
		return fmt.Errorf("dialing SSH, %w", err)
	}

	// add our ssh key to authorized keys so it will last longer than 60 seconds
	session, err := client.NewSession()
	if err != nil {
		return fmt.Errorf("creating SSH session, %w", err)
	}

	_, err = session.CombinedOutput(fmt.Sprintf("echo '%s' >> ~/.ssh/authorized_keys", string(testInstance.sshKey.public)))
	if err != nil {
		return fmt.Errorf("registering SSH key, %w", err)
	}

	// write our private SSH key to disk and register it
	f, err := os.CreateTemp("", ".ssh-key-*")
	if err != nil {
		return fmt.Errorf("creating SSH key, %w", err)
	}
	sshKeyFile := f.Name()
	if err = os.Chmod(sshKeyFile, 0400); err != nil {
		return fmt.Errorf("chmod'ing SSH key, %w", err)
	}

	if _, err = f.Write(testInstance.sshKey.private); err != nil {
		return fmt.Errorf("writing SSH key, %w", err)
	}
	AddSSHKey(testInstance.instanceID, sshKeyFile)
	testInstance.sshPublicKeyFile = sshKeyFile
	return nil
}

func (a *AWSRunner) launchNewInstance(img internalAWSImage) (*ec2.Instance, error) {
	input := &ec2.RunInstancesInput{
		InstanceType: &img.instanceType,
		ImageId:      &img.amiID,
		MinCount:     aws.Int64(1),
		MaxCount:     aws.Int64(1),
		NetworkInterfaces: []*ec2.InstanceNetworkInterfaceSpecification{
			{
				AssociatePublicIpAddress: aws.Bool(true),
				DeviceIndex:              aws.Int64(0),
			},
		},
		TagSpecifications: []*ec2.TagSpecification{
			{
				ResourceType: aws.String(ec2.ResourceTypeInstance),
				Tags: []*ec2.Tag{
					{
						Key:   aws.String("Name"),
						Value: aws.String(a.cfg.InstanceNamePrefix + img.imageDesc),
					},
					// tagged so we can find it easily
					{
						Key:   aws.String(amiIDTag),
						Value: aws.String(img.amiID),
					},
				},
			},
			{
				ResourceType: aws.String(ec2.ResourceTypeVolume),
				Tags: []*ec2.Tag{
					{
						Key:   aws.String("Name"),
						Value: aws.String(a.cfg.InstanceNamePrefix + img.imageDesc),
					},
				},
			},
		},
		BlockDeviceMappings: []*ec2.BlockDeviceMapping{
			{
				DeviceName: aws.String("/dev/xvda"),
				Ebs: &ec2.EbsBlockDevice{
					VolumeSize: aws.Int64(20),
					VolumeType: aws.String("gp3"),
				},
			},
		},
	}
	if len(img.userData) > 0 {
		input.UserData = aws.String(base64.StdEncoding.EncodeToString(img.userData))
	}
	if img.instanceProfile != "" {
		input.IamInstanceProfile = &ec2.IamInstanceProfileSpecification{
			Name: &img.instanceProfile,
		}
	}

	rsv, err := a.ec2Service.RunInstances(input)
	if err != nil {
		return nil, fmt.Errorf("creating instance, %w", err)
	}

	return rsv.Instances[0], nil
}

func (a *AWSRunner) getSSMImage(path string) (string, error) {
	rsp, err := a.ssmService.GetParameter(&ssm.GetParameterInput{
		Name: &path,
	})
	if err != nil {
		return "", fmt.Errorf("getting AMI ID from SSM path %q, %w", path, err)
	}
	return *rsp.Parameter.Value, nil
}

type awsInstance struct {
	instance         *ec2.Instance
	instanceID       string
	sshKey           *temporarySSHKey
	publicIP         string
	sshPublicKeyFile string
}

type temporarySSHKey struct {
	public  []byte
	private []byte
	signer  ssh.Signer
}

func generateSSHKeypair() (*temporarySSHKey, error) {
	privateKey, err := rsa.GenerateKey(crand.Reader, 2048)
	if err != nil {
		return nil, fmt.Errorf("generating private key, %w", err)
	}
	if err := privateKey.Validate(); err != nil {
		return nil, fmt.Errorf("validating private key, %w", err)
	}

	pubSSH, err := ssh.NewPublicKey(&privateKey.PublicKey)
	if err != nil {
		return nil, fmt.Errorf("creating SSH key, %w", err)
	}
	pubKey := ssh.MarshalAuthorizedKey(pubSSH)

	privDER := x509.MarshalPKCS1PrivateKey(privateKey)
	privBlock := pem.Block{
		Type:    "RSA PRIVATE KEY",
		Headers: nil,
		Bytes:   privDER,
	}
	privatePEM := pem.EncodeToMemory(&privBlock)

	signer, err := ssh.NewSignerFromKey(privateKey)
	if err != nil {
		return nil, fmt.Errorf("creating signer, %w", err)
	}
	return &temporarySSHKey{
		public:  pubKey,
		private: privatePEM,
		signer:  signer,
	}, nil
}
