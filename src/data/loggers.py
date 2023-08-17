from colorama import Fore, Style, Back

def log_success(msg):
    """
    Logs success in bold green
    """
    print(Style.BRIGHT + Fore.GREEN + msg + Style.RESET_ALL + Fore.RESET)


def log_failure(msg):
    """
    Logs failure in bold red
    """
    print(Style.BRIGHT + Fore.RED)
    print(msg)
    print(Style.RESET_ALL + Fore.RESET)


def log_small(msg):
    print(Style.DIM)
    print('--> ', msg)
    print(Style.RESET_ALL)