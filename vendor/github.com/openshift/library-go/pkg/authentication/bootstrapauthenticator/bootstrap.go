package bootstrapauthenticator

import (
	"context"
	"crypto/sha512"
	"encoding/base64"
	"fmt"
	"time"

	"golang.org/x/crypto/bcrypt"
	"k8s.io/klog/v2"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apiserver/pkg/authentication/authenticator"
	"k8s.io/apiserver/pkg/authentication/user"
	v1 "k8s.io/client-go/kubernetes/typed/core/v1"
)

const (
	// BootstrapUser is the magic bootstrap OAuth user that can perform any action
	BootstrapUser = "kube:admin"
	// support basic auth which does not allow : in username
	bootstrapUserBasicAuth = "kubeadmin"
	// force the use of a secure password length
	// expected format is 5char-5char-5char-5char
	minPasswordLen = 23
)

var (
	// make it obvious that we refuse to honor short passwords
	errPasswordTooShort = fmt.Errorf("%s password must be at least %d characters long", bootstrapUserBasicAuth, minPasswordLen)

	// we refuse to honor a secret that is too new when compared to kube-system
	// since kube-system always exists and cannot be deleted
	// and creation timestamp is controlled by the api, we can use this to
	// detect if the secret was recreated after the initial bootstrapping
	errSecretRecreated = fmt.Errorf("%s secret cannot be recreated", bootstrapUserBasicAuth)
)

// Password checks a username and password against a backing authentication
// store and returns a Response or an error if the password could not be
// checked.
//
// This was copied from
// k8s.io/apiserver/pkg/authentication/authenticator due to its
// removal in 1.19.
type Password interface {
	AuthenticatePassword(ctx context.Context, user, password string) (*authenticator.Response, bool, error)
}

func New(getter BootstrapUserDataGetter) Password {
	return &bootstrapPassword{
		getter: getter,
		names:  sets.NewString(BootstrapUser, bootstrapUserBasicAuth),
	}
}

type bootstrapPassword struct {
	getter BootstrapUserDataGetter
	names  sets.String
}

func (b *bootstrapPassword) AuthenticatePassword(ctx context.Context, username, password string) (*authenticator.Response, bool, error) {
	if !b.names.Has(username) {
		return nil, false, nil
	}

	data, ok, err := b.getter.Get()
	if err != nil || !ok {
		return nil, ok, err
	}

	// check length after we know that the secret is functional since
	// we do not want to complain when the bootstrap user is disabled
	if len(password) < minPasswordLen {
		return nil, false, errPasswordTooShort
	}

	if err := bcrypt.CompareHashAndPassword(data.PasswordHash, []byte(password)); err != nil {
		if err == bcrypt.ErrMismatchedHashAndPassword {
			klog.V(4).Infof("%s password mismatch", bootstrapUserBasicAuth)
			return nil, false, nil
		}
		return nil, false, err
	}

	// do not set other fields, see identitymapper.userToInfo func
	return &authenticator.Response{
		User: &user.DefaultInfo{
			Name: BootstrapUser,
			UID:  data.UID, // uid ties this authentication to the current state of the secret
		},
	}, true, nil
}

type BootstrapUserData struct {
	PasswordHash []byte
	UID          string
}

type BootstrapUserDataGetter interface {
	Get() (data *BootstrapUserData, ok bool, err error)
	IsEnabled() (bool, error)
}

func NewBootstrapUserDataGetter(secrets v1.SecretsGetter, namespaces v1.NamespacesGetter) BootstrapUserDataGetter {
	return &bootstrapUserDataGetter{
		secrets:    secrets.Secrets(metav1.NamespaceSystem),
		namespaces: namespaces.Namespaces(),
	}
}

type bootstrapUserDataGetter struct {
	secrets    v1.SecretInterface
	namespaces v1.NamespaceInterface
}

func (b *bootstrapUserDataGetter) Get() (*BootstrapUserData, bool, error) {
	secret, err := b.getBootstrapUserSecret()
	if err != nil || secret == nil {
		return nil, false, err
	}

	hashedPassword := secret.Data[bootstrapUserBasicAuth]

	// make sure the value is a valid bcrypt hash
	if _, err := bcrypt.Cost(hashedPassword); err != nil {
		return nil, false, err
	}

	exactSecret := string(secret.UID) + secret.ResourceVersion
	both := append([]byte(exactSecret), hashedPassword...)

	// use a hash to avoid leaking any derivative of the password
	// this makes it easy for us to tell if the secret changed
	uidBytes := sha512.Sum512(both)

	return &BootstrapUserData{
		PasswordHash: hashedPassword,
		UID:          base64.RawURLEncoding.EncodeToString(uidBytes[:]),
	}, true, nil
}

func (b *bootstrapUserDataGetter) IsEnabled() (bool, error) {
	secret, err := b.getBootstrapUserSecret()
	if err == errSecretRecreated {
		return false, nil
	}
	if err != nil || secret == nil {
		return false, err
	}
	return true, nil
}

func (b *bootstrapUserDataGetter) getBootstrapUserSecret() (*corev1.Secret, error) {
	secret, err := b.secrets.Get(context.TODO(), bootstrapUserBasicAuth, metav1.GetOptions{})
	if errors.IsNotFound(err) {
		klog.V(4).Infof("%s secret does not exist", bootstrapUserBasicAuth)
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	if secret.DeletionTimestamp != nil {
		klog.V(4).Infof("%s secret is being deleted", bootstrapUserBasicAuth)
		return nil, nil
	}
	namespace, err := b.namespaces.Get(context.TODO(), metav1.NamespaceSystem, metav1.GetOptions{})
	if err != nil {
		return nil, err
	}
	if secret.CreationTimestamp.After(namespace.CreationTimestamp.Add(time.Hour)) {
		return nil, errSecretRecreated
	}
	return secret, nil
}
