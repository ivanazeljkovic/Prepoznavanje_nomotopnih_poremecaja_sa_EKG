# Prepoznavanje nomotopnih poremećaja sa EKG snimaka
<b>Problem koji se rešava:</b> Prepoznavanje jedne od 3 vrste nomotopnih poremećaja u srčanom ritmu sa EKG snimaka: 
- Sinusna tahikardija
- Sinusna bradikardija
- Sinusna aritmija

U svrhu rešavanja problema izvršena je detaljna analiza slike na kojoj se nalazi EKG snimak. <br><br>
<b> Analiza slike: </b> 
<ul>
  <li> K-means algoritam u svrhu izdvajanja EKG signala sa slike </li>
  <li> Manipulacija prostora boja u svrhu izdvajanja mreže na kojoj je predstavljen EKG signal </li> 
  <li> Pronalaženje koordinata vertikalnih linija, početka i kraja svakog R-R intervala, 
  sredina svakog P-QRS-T kompleksa, kao i vrha svakog od R zubaca </li>
</ul>


<b>Tehnologije i alati:</b>
<ul>
  <li> Programski jezik: Python 3.4 </li>
  <li> Razvojno okruženje: PyCharm IDE </li>
  <li> Biblioteka za obradu slike: OpenCV 3.2 </li>
</ul>


<b> Pokretanje aplikacije: </b>
<ol type='number'>
  <li> Klonirati GitHub repozitorijum lokalno/Skinuti sadržaj kao .zip arhivu </li>
  <li> U skripti main.py na liniji 316 upisati naziv slike (iz Input-images foldera) za koju se vrši analiza </li>
  <li> Opciono: obrisati pomoćni sadržaj izgenerisan aplikacijom - sadržaj foldera Output-images </li>
  <li> Aplikaciju pokrenuti konzolno komandom: python main.py iz korespodentnog direktorijuma ili upotrebom nekog od razvojnih okruženja </li>
</ol>


