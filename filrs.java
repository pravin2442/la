public class AdditiveCipher {
    public static String encrypt(String text, int key) {
        text = text.toUpperCase();
        StringBuilder result = new StringBuilder();

        for (char c : text.toCharArray()) {
            if (Character.isLetter(c)) {
                char ch = (char) ((c - 'A' + key) % 26 + 'A');
                result.append(ch);
            } else result.append(c);
        }
        return result.toString();
    }

    public static String decrypt(String text, int key) {
        return encrypt(text, 26 - key);
    }

    public static void main(String[] args) {
        String plain = "HELLO";
        int key = 5;
        String cipher = encrypt(plain, key);
        String decrypted = decrypt(cipher, key);

        System.out.println("Plaintext: " + plain);
        System.out.println("Encrypted: " + cipher);
        System.out.println("Decrypted: " + decrypted);
    }
}


public class MultiplicativeCipher {
    public static int modInverse(int key) {
        for (int i = 1; i < 26; i++) {
            if ((key * i) % 26 == 1) return i;
        }
        return -1;
    }

    public static String encrypt(String text, int key) {
        text = text.toUpperCase();
        StringBuilder result = new StringBuilder();
        for (char c : text.toCharArray()) {
            if (Character.isLetter(c)) {
                int x = (c - 'A');
                char enc = (char) ((x * key) % 26 + 'A');
                result.append(enc);
            } else result.append(c);
        }
        return result.toString();
    }

    public static String decrypt(String text, int key) {
        int inv = modInverse(key);
        StringBuilder result = new StringBuilder();
        for (char c : text.toCharArray()) {
            if (Character.isLetter(c)) {
                int x = (c - 'A');
                char dec = (char) ((x * inv) % 26 + 'A');
                result.append(dec);
            } else result.append(c);
        }
        return result.toString();
    }

    public static void main(String[] args) {
        String plain = "HELLO";
        int key = 5;
        String cipher = encrypt(plain, key);
        String decrypted = decrypt(cipher, key);
        System.out.println("Plaintext: " + plain);
        System.out.println("Encrypted: " + cipher);
        System.out.println("Decrypted: " + decrypted);
    }
}


public class CaesarCipher {
    public static String encrypt(String text) {
        int shift = 3;
        StringBuilder result = new StringBuilder();

        for (char c : text.toUpperCase().toCharArray()) {
            if (Character.isLetter(c)) {
                result.append((char) ((c - 'A' + shift) % 26 + 'A'));
            } else result.append(c);
        }
        return result.toString();
    }

    public static String decrypt(String text) {
        int shift = 3;
        return encrypt(text, 26 - shift);
    }

    private static String encrypt(String text, int key) {
        StringBuilder result = new StringBuilder();
        for (char c : text.toUpperCase().toCharArray()) {
            if (Character.isLetter(c)) {
                result.append((char) ((c - 'A' + key) % 26 + 'A'));
            } else result.append(c);
        }
        return result.toString();
    }

    public static void main(String[] args) {
        String plain = "HELLO";
        String cipher = encrypt(plain);
        String decrypted = decrypt(cipher);

        System.out.println("Plaintext: " + plain);
        System.out.println("Encrypted: " + cipher);
        System.out.println("Decrypted: " + decrypted);
    }
}


public class AffineCipher {
    static int modInverse(int a) {
        for (int i = 1; i < 26; i++) {
            if ((a * i) % 26 == 1) return i;
        }
        return -1;
    }

    static String encrypt(String text, int a, int b) {
        StringBuilder res = new StringBuilder();
        for (char c : text.toUpperCase().toCharArray()) {
            if (Character.isLetter(c))
                res.append((char) (((a * (c - 'A') + b) % 26) + 'A'));
            else res.append(c);
        }
        return res.toString();
    }

    static String decrypt(String cipher, int a, int b) {
        int inv = modInverse(a);
        StringBuilder res = new StringBuilder();
        for (char c : cipher.toUpperCase().toCharArray()) {
            if (Character.isLetter(c))
                res.append((char) (((inv * ((c - 'A' - b + 26)) % 26) + 'A')));
            else res.append(c);
        }
        return res.toString();
    }

    public static void main(String[] args) {
        String plain = "HELLO";
        int a = 5, b = 8;
        String cipher = encrypt(plain, a, b);
        String decrypted = decrypt(cipher, a, b);
        System.out.println("Plaintext: " + plain);
        System.out.println("Encrypted: " + cipher);
        System.out.println("Decrypted: " + decrypted);
    }
}


public class VigenereCipher {
    static String generateKey(String text, String key) {
        StringBuilder newKey = new StringBuilder(key);
        while (newKey.length() < text.length())
            newKey.append(key);
        return newKey.substring(0, text.length()).toUpperCase();
    }

    static String encrypt(String text, String key) {
        StringBuilder result = new StringBuilder();
        for (int i = 0; i < text.length(); i++) {
            char c = (char) (((text.charAt(i) - 'A' + key.charAt(i) - 'A') % 26) + 'A');
            result.append(c);
        }
        return result.toString();
    }

    static String decrypt(String cipher, String key) {
        StringBuilder result = new StringBuilder();
        for (int i = 0; i < cipher.length(); i++) {
            char c = (char) (((cipher.charAt(i) - key.charAt(i) + 26) % 26) + 'A');
            result.append(c);
        }
        return result.toString();
    }

    public static void main(String[] args) {
        String plain = "ATTACK";
        String key = "LEMON";
        key = generateKey(plain, key);
        String cipher = encrypt(plain, key);
        String decrypted = decrypt(cipher, key);
        System.out.println("Plaintext: " + plain);
        System.out.println("Encrypted: " + cipher);
        System.out.println("Decrypted: " + decrypted);
    }
}


public class HillCipher {
    static int[][] key = {{3, 3}, {2, 5}}; // example key

    static String encrypt(String msg) {
        msg = msg.toUpperCase();
        int[] P = {msg.charAt(0) - 'A', msg.charAt(1) - 'A'};
        int[] C = new int[2];
        C[0] = (key[0][0] * P[0] + key[0][1] * P[1]) % 26;
        C[1] = (key[1][0] * P[0] + key[1][1] * P[1]) % 26;
        return "" + (char)(C[0] + 'A') + (char)(C[1] + 'A');
    }

    public static void main(String[] args) {
        String plain = "HI";
        String cipher = encrypt(plain);
        System.out.println("Plaintext: " + plain);
        System.out.println("Encrypted: " + cipher);
        // Decryption skipped for simplicity
    }
}

public class SimpleHillCipher {
    // Key matrix (2x2)
    static int[][] key = { {3, 3}, {2, 5} };

    // Find determinant
    static int determinant(int[][] m) {
        return (m[0][0] * m[1][1] - m[0][1] * m[1][0]);
    }

    // Find modular inverse
    static int modInverse(int a, int m) {
        a = a % m;
        for (int x = 1; x < m; x++) {
            if ((a * x) % m == 1)
                return x;
        }
        return 1;
    }

    // Encrypt function
    static String encrypt(String text) {
        text = text.toUpperCase();
        if (text.length() % 2 != 0)
            text += "X"; // make even length
        StringBuilder result = new StringBuilder();

        for (int i = 0; i < text.length(); i += 2) {
            int p1 = text.charAt(i) - 'A';
            int p2 = text.charAt(i + 1) - 'A';

            int c1 = (key[0][0] * p1 + key[0][1] * p2) % 26;
            int c2 = (key[1][0] * p1 + key[1][1] * p2) % 26;

            result.append((char) (c1 + 'A'));
            result.append((char) (c2 + 'A'));
        }
        return result.toString();
    }

    // Decrypt function
    static String decrypt(String cipher) {
        cipher = cipher.toUpperCase();
        StringBuilder result = new StringBuilder();

        int det = determinant(key);
        int invDet = modInverse(det, 26);

        // inverse key matrix mod 26
        int[][] invKey = new int[2][2];
        invKey[0][0] = (key[1][1] * invDet) % 26;
        invKey[1][1] = (key[0][0] * invDet) % 26;
        invKey[0][1] = (-key[0][1] * invDet) % 26;
        invKey[1][0] = (-key[1][0] * invDet) % 26;

        // fix negatives
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 2; j++)
                invKey[i][j] = (invKey[i][j] + 26) % 26;

        for (int i = 0; i < cipher.length(); i += 2) {
            int c1 = cipher.charAt(i) - 'A';
            int c2 = cipher.charAt(i + 1) - 'A';

            int p1 = (invKey[0][0] * c1 + invKey[0][1] * c2) % 26;
            int p2 = (invKey[1][0] * c1 + invKey[1][1] * c2) % 26;

            result.append((char) (p1 + 'A'));
            result.append((char) (p2 + 'A'));
        }
        return result.toString();
    }

    public static void main(String[] args) {
        String text = "HI";
        System.out.println("Plaintext: " + text);

        String cipher = encrypt(text);
        System.out.println("Encrypted: " + cipher);

        String decrypted = decrypt(cipher);
        System.out.println("Decrypted: " + decrypted);
    }
}


import java.math.BigInteger;
import java.nio.charset.StandardCharsets;
import java.security.SecureRandom;

public class SimpleRSA {
    public static void main(String[] args) throws Exception {
        SecureRandom rnd = new SecureRandom();

        // 1) Generate two primes p and q (small for demo; in real life use 1024+ bits)
        BigInteger p = BigInteger.probablePrime(64, rnd);
        BigInteger q = BigInteger.probablePrime(64, rnd);

        BigInteger n = p.multiply(q);                // modulus
        BigInteger phi = p.subtract(BigInteger.ONE).multiply(q.subtract(BigInteger.ONE));

        BigInteger e = BigInteger.valueOf(65537);    // common public exponent
        if (!e.gcd(phi).equals(BigInteger.ONE)) {
            // fallback if not coprime (rare with small primes)
            e = BigInteger.valueOf(3);
            while (!e.gcd(phi).equals(BigInteger.ONE)) e = e.add(BigInteger.TWO);
        }

        BigInteger d = e.modInverse(phi);            // private exponent

        System.out.println("Public key (n, e): n=" + n + " e=" + e);
        System.out.println("Private key d=" + d);

        // Message to encrypt
        String message = "hello";
        byte[] messageBytes = message.getBytes(StandardCharsets.UTF_8);
        BigInteger m = new BigInteger(1, messageBytes); // convert to positive BigInteger

        // Encrypt: c = m^e mod n
        BigInteger c = m.modPow(e, n);
        System.out.println("Ciphertext: " + c);

        // Decrypt: m = c^d mod n
        BigInteger m2 = c.modPow(d, n);
        byte[] decryptedBytes = m2.toByteArray();

        // BigInteger may include leading zero byte to indicate sign — fix for exact original bytes:
        if (decryptedBytes[0] == 0) {
            byte[] tmp = new byte[decryptedBytes.length - 1];
            System.arraycopy(decryptedBytes, 1, tmp, 0, tmp.length);
            decryptedBytes = tmp;
        }

        String decrypted = new String(decryptedBytes, StandardCharsets.UTF_8);
        System.out.println("Decrypted message: " + decrypted);
    }
}

import java.math.BigInteger;
import java.security.SecureRandom;

public class SimpleDH {
    public static void main(String[] args) {
        SecureRandom rnd = new SecureRandom();

        // For demo use a safe prime; here we use a small prime. Real world: use 2048-bit prime groups.
        BigInteger p = new BigInteger("104729"); // example prime (you can pick larger)
        BigInteger g = BigInteger.valueOf(2);

        // Each party picks a private key
        BigInteger a = new BigInteger(64, rnd); // Alice private
        BigInteger b = new BigInteger(64, rnd); // Bob private

        // Public values
        BigInteger A = g.modPow(a, p); // Alice -> send A
        BigInteger B = g.modPow(b, p); // Bob -> send B

        // Shared secrets computed locally
        BigInteger secretA = B.modPow(a, p);
        BigInteger secretB = A.modPow(b, p);

        System.out.println("p=" + p + " g=" + g);
        System.out.println("Alice public A=" + A);
        System.out.println("Bob public B=" + B);
        System.out.println("Alice shared secret: " + secretA);
        System.out.println("Bob shared secret:   " + secretB);
        System.out.println("Shared equal? " + secretA.equals(secretB));
    }
}


import java.math.BigInteger;
import java.nio.charset.StandardCharsets;
import java.security.SecureRandom;

public class SimpleElGamal {
    public static void main(String[] args) {
        SecureRandom rnd = new SecureRandom();

        // For demo we use a small prime; in practice use large safe primes
        BigInteger p = new BigInteger("467"); // prime
        BigInteger g = BigInteger.valueOf(2);

        // Key generation
        BigInteger x = new BigInteger(64, rnd).mod(p.subtract(BigInteger.TWO)).add(BigInteger.ONE); // private
        BigInteger y = g.modPow(x, p); // public y = g^x mod p

        System.out.println("p=" + p + " g=" + g);
        System.out.println("Private x=" + x);
        System.out.println("Public y=" + y);

        // Message (convert to integer < p)
        String message = "42"; // for simplicity keep message as number or short text < p
        // If using bytes->BigInteger ensure number < p; here we use small numeric message for clarity
        BigInteger m = new BigInteger(message);

        // Encryption: pick random k
        BigInteger k = new BigInteger(64, rnd).mod(p.subtract(BigInteger.TWO)).add(BigInteger.ONE);
        BigInteger c1 = g.modPow(k, p);
        BigInteger c2 = m.multiply(y.modPow(k, p)).mod(p);

        System.out.println("Ciphertext (c1, c2): " + c1 + " , " + c2);

        // Decryption: m = c2 * (c1^x)^-1 mod p
        BigInteger s = c1.modPow(x, p);
        BigInteger sInv = s.modInverse(p);
        BigInteger recovered = c2.multiply(sInv).mod(p);

        System.out.println("Decrypted message: " + recovered);
    }
}


//playfair cipher

import java.util.*;

public class PlayfairCipher {

    private static char[][] keyMatrix = new char[5][5];

    // Generate key matrix
    public static void generateKeyMatrix(String key) {
        key = key.toUpperCase().replaceAll("[^A-Z]", "").replace('J', 'I');
        boolean[] used = new boolean[26];
        int k = 0;

        // Fill matrix with key letters
        for (char c : key.toCharArray()) {
            int idx = c - 'A';
            if (!used[idx]) {
                keyMatrix[k / 5][k % 5] = c;
                used[idx] = true;
                k++;
            }
        }

        // Fill remaining letters
        for (char c = 'A'; c <= 'Z'; c++) {
            if (c == 'J') continue;
            int idx = c - 'A';
            if (!used[idx]) {
                keyMatrix[k / 5][k % 5] = c;
                k++;
            }
        }
    }

    // Find position of character in key matrix
    public static int[] findPosition(char c) {
        if (c == 'J') c = 'I';
        for (int i = 0; i < 5; i++)
            for (int j = 0; j < 5; j++)
                if (keyMatrix[i][j] == c) return new int[]{i, j};
        return null;
    }

    // Prepare text (add X if duplicate letters in a digraph)
    public static String prepareText(String text, boolean encrypt) {
        text = text.toUpperCase().replaceAll("[^A-Z]", "").replace('J', 'I');
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < text.length(); i++) {
            char a = text.charAt(i);
            char b = (i + 1 < text.length()) ? text.charAt(i + 1) : 'X';
            sb.append(a);
            if (encrypt) {
                if (a == b) sb.append('X');
                else sb.append(b);
                i++;
            }
        }
        if (sb.length() % 2 != 0) sb.append('X');
        return sb.toString();
    }

    // Encrypt or Decrypt a digraph
    public static String process(String text, boolean encrypt) {
        StringBuilder result = new StringBuilder();
        int shift = encrypt ? 1 : 4; // move right for encryption, left for decryption (mod 5)
        for (int i = 0; i < text.length(); i += 2) {
            char a = text.charAt(i);
            char b = text.charAt(i + 1);
            int[] posA = findPosition(a);
            int[] posB = findPosition(b);

            if (posA[0] == posB[0]) { // same row
                result.append(keyMatrix[posA[0]][(posA[1] + shift) % 5]);
                result.append(keyMatrix[posB[0]][(posB[1] + shift) % 5]);
            } else if (posA[1] == posB[1]) { // same column
                result.append(keyMatrix[(posA[0] + shift) % 5][posA[1]]);
                result.append(keyMatrix[(posB[0] + shift) % 5][posB[1]]);
            } else { // rectangle
                result.append(keyMatrix[posA[0]][posB[1]]);
                result.append(keyMatrix[posB[0]][posA[1]]);
            }
        }
        return result.toString();
    }

    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        System.out.print("Enter key: ");
        String key = sc.nextLine();
        generateKeyMatrix(key);

        System.out.println("Key Matrix:");
        for (char[] row : keyMatrix) System.out.println(Arrays.toString(row));

        System.out.print("Enter plaintext: ");
        String plaintext = sc.nextLine();

        String preparedText = prepareText(plaintext, true);
        String ciphertext = process(preparedText, true);
        System.out.println("Encrypted Text: " + ciphertext);

        String decryptedText = process(ciphertext, false);
        System.out.println("Decrypted Text: " + decryptedText);
    }
}
