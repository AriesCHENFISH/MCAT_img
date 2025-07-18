# main.py
import sys

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python main.py [pretrain|extract]")
        exit(1)

    mode = sys.argv[1].lower()

    if mode == 'pretrain_ct':
        import scripts.mae_ct_pretrain
    elif mode == 'extract_ct':
        import scripts.feature_ct_extract
    elif mode == 'pretrain_dsa':
        # import scripts.mae_dsa_pretrain
        import scripts.resnet_dsa_pretrain
    elif mode == 'extract_dsa':
        import scripts.feature_dsa_extract

    else:
        print(f"Unknown mode: {mode}")
        print("Usage: python main.py [pretrain|extract]")
