import random, string

def gen_code(N, W_ID=None):
    if W_ID is None :
        """Generate random completion code."""
        return ''.join(random.choices(string.ascii_lowercase + string.digits, k=N))
    else : 
        """Generate random completion code + W_ID."""
        return W_ID + '_' + ''.join(random.choices(string.ascii_lowercase + string.digits, k=N//4))
