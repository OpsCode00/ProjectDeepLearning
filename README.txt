MIGLIORI RUN
MODELLO 32x32:
run32x32_BS=128_LR=0.001_EPOCHS=30_AUG=False_TRA=False_OPT=ADAM_1 Buona run, appena oscillante la loss
run32x32_BS=128_LR=0.0005_EPOCHS=30_AUG=False_TRA=False_OPT=ADAM Non buonissima

Migliore con SGD
run32x32_BS=128_LR=0.01_MOM=0.9_EPOCHS=30_AUG=False_TRA=False_OPT=SGD Migliore run con questo modello

run32x32_BS=128_LR=0.005_MOM=0.9_EPOCHS=30_AUG=False_TRA=False_OPT=SGD 94,50 max acc val inizio overfitting più marcato
run32x32_BS=128_LR=0.005_MOM=0.9_EPOCHS=30_AUG=False_TRA=False_OPT=SGD_1 0,7 validation split
run32x32_BS=128_LR=0.003_MOM=0.9_EPOCHS=30_AUG=False_TRA=False_OPT=SGD 93,93 max acc val inizio overfitting
run32x32_BS=128_LR=0.0025_MOM=0.9_EPOCHS=20_AUG=False_TRA=False_OPT=SGD no overfitting
run32x32_BS=128_LR=0.002_MOM=0.9_EPOCHS=30_AUG=False_TRA=False_OPT=SGD 93,23 max acc val inizio overfitting
run32x32_BS=256_LR=0.002_MOM=0.9_EPOCHS=30_AUG=False_TRA=False_OPT=SGD 93,23 max acc val inizio overfitting
run32x32_BS=64_LR=0.002_MOM=0.9_EPOCHS=30_AUG=False_TRA=False_OPT=SGD 93,23 max acc val inizio overfitting
run32x32_BS=128_LR=0.001_MOM=0.9_EPOCHS=30_AUG=False_TRA=False_OPT=SGD_1 91,6 max acc val no overfitting 
run32x32_BS=128_LR=0.001_MOM=0.9_EPOCHS=30_AUG=False_TRA=False_OPT=SGD 91,6 max acc val no overfitting
run32x32_BS=128_LR=0.01_EPOCHS=30_AUG=False_TRA=False_OPT=ADAM val loss molto oscillante, accuratezza ancora bassa
****AUGMENTATION****
run32x32_BS=128_LR=0.001_EPOCHS=30_AUG=True_TRA=False_OPT=ADAM Anche in questo caso fa più fatica in training, ma buono
run32x32_BS=128_LR=0.0008_EPOCHS=30_AUG=True_TRA=False_OPT=ADAM Buono ma ha smesso di decrescere la loss

BATCH NORMALIZATION
SCHEDULER COSINE ANNEALING
run32x32BN_BS=128_LR=0.001_EPOCHS=30_AUG=False_TRA=False_OPT=ADAM_6 
non funzionanti
run32x32BN_BS=128_LR=0.001_EPOCHS=30_AUG=False_TRA=False_OPT=ADAM Overfittato ma ottenuto migliori performance
run32x32BN_BS=128_LR=0.001_EPOCHS=30_AUG=False_TRA=False_OPT=ADAM_1 Overfittato ottenuti risultati appena inferiori T_max=10
SCHEDULER STEPLR
run32x32BN_BS=128_LR=0.001_EPOCHS=30_AUG=False_TRA=False_OPT=ADAM_4
run32x32BN_BS=128_LR=0.001_EPOCHS=30_AUG=False_TRA=False_OPT=ADAM_5
non funzionanti
run32x32BN_BS=128_LR=0.001_EPOCHS=30_AUG=False_TRA=False_OPT=ADAM_2 non funzionante
run32x32BN_BS=128_LR=0.001_EPOCHS=30_AUG=False_TRA=False_OPT=ADAM_3 non funzionante

MODELLO 28x28:
log_bs=128_lr=0.001_e=25_m=0.9_aug=False_tra=False poche epoche, loss ancora in discesa 93,9 max acc val
log_bs=128_lr=0.003_e=25_m=0.9_aug=False_tra=False poche epoche, loss ancora in discesa 93,9 max acc val
log_bs=128_lr=0.003_e=25_m=0.9_aug=False_tra=False overfit medio, 94,16 max acc val
log_bs=128_lr=0.01_e=25_m=0.9_aug=False_tra=False overfit medio, 95,17 max acc val
log_bs=128_lr=0.01_e=30_m=0.9_aug=False_tra=False_opt=SGD leggero overfit, loss val molto oscillante ma in discesa 96 max acc val
log_bs=128_lr=0.01_e=50_m=0.9_aug=False_tra=False forse overfit, loss val oscillante 95,7 max acc val

MODELLO MLP:
runMLP_BS=128_LR=0.00012_MOM=0.9_EPOCHS=15_AUG=False_TRA=False_OPT=SGD Migliore con SGD
runMLP_BS=128_LR=2.5e-05_EPOCHS=15_AUG=False_TRA=False_OPT=ADAM Migliore con ADAM
runMLP_BS=128_LR=0.0001_MOM=0.9_EPOCHS=15_AUG=False_TRA=False_OPT=SGD
***AUGMENTATION le immagini di training possono essere più difficili da classificare
rispetto alle immagini di validazione. Ciò aumenta la loss e riduce l'accuracy durante il training.***
runMLP_BS=128_LR=0.001_MOM=0.9_EPOCHS=15_AUG=True_TRA=False_OPT=SGD Questa run con solo 0.3 del dataset augmentato, meglio
runMLP_BS=128_LR=2.5e-05_EPOCHS=15_AUG=True_TRA=False_OPT=ADAM Idem per sopra, si comporta meglio quindi era troppo potente la data augmentation
Con gli stessi iperparametri ma il dataset augmentato si ottengono comportamenti strani
runMLP_BS=128_LR=2.5e-05_EPOCHS=15_AUG=True_TRA=False_OPT=ADAM 
runMLP_BS=128_LR=0.00012_MOM=0.9_EPOCHS=15_AUG=True_TRA=False_OPT=SGD 