import numpy as np
import cv2
import math


frequency = [300, 150, 100, 75, 60, 50]
#Sinusna bradikardija -> frekvencija < 60
#Sinusna tahikardija  -> 100 < frekvencija < 140
#Sinusna (respiratorna) aritmija -> Naizmenicni periodi tahikardije i bradikardije

def load_image(name):
    return cv2.imread(name + '.jpg')



def k_means_signal(name):
    img = load_image(name)
    Z = img.reshape((-1, 3))
    Z = np.float32(Z)

    # definisanje kriterijuma prestanka, broja klastera za K-means
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 2
    ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))

    #cv2.imshow('ECG', res2)
    # na slici se prikazuje signal koji je izdvojen iz originalne slike upotrebom klasterizacije (K-means sa 2 klastera)
    cv2.imwrite('K-means-signal.jpg', res2)



def get_grid(name):
    lower_range_grid = np.array([50,0,170], dtype=np.uint8)
    upper_range_grid = np.array([255,255,250], dtype=np.uint8)

    img = load_image(name)
    # setovanje prostora boja tako da se identifikuje mreza sa originalne slike
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    mask_grid = cv2.inRange(hsv, lower_range_grid, upper_range_grid)
    mask_grid_inv = cv2.bitwise_not(mask_grid)

    #cv2.imshow('Grid', mask_grid_inv)
    # na slici se prikazuje mreza koja je izdvojena iz originalne slike
    cv2.imwrite('grid.jpg', mask_grid_inv)



def get_vertical_lines():
    img = cv2.imread('grid.jpg')
    height, width, depth = img.shape
    x_coordinates = []

    # pronalazenje x koordinata vertikalnih linija u mrezi
    for pixel in range(width):
        pixel_rgb = img[1, pixel]
        if(pixel_rgb[0] < 10 and pixel_rgb[1] < 10 and pixel_rgb[2] < 10):
            #cv2.circle(img, (pixel, 1), 1, (0,0,255), thickness=2, lineType=8, shift=0)
            x_coordinates.append(pixel)

    #cv2.imshow('Vertical lines', img)
    return x_coordinates



def get_boundaries_of_complex(row_height):
    # u metodi se traze granice prostora izmedju 2 R zupca susednih P-QRS-T kompleksa
    # tako sto se prolazi kroz pixele na y koordinati (prostor u 2. vrsti mreze, pocevsi od vrha iste)
    y = math.floor(row_height*1.5)
    black_spaces = []
    boundaries_complex = []

    img = cv2.imread('K-means-signal.jpg')
    img = cv2.bitwise_not(img)
    height, width, depth = img.shape


    previous_pixel_black = False
    for pixel in range(width):
        pixel_rgb = img[y, pixel]
        if (pixel_rgb[0] < 20 and pixel_rgb[1] < 20 and pixel_rgb[2] < 20):
            # ukoliko smo naisli na prvi taman pixel, oznacavamo pocetak novog tamnog prostora
            if not previous_pixel_black:
                previous_pixel_black = True
                black_spaces.append([pixel, 0])
        else:
            # u suprotnom, oznacavamo kraj tekuceg tamnog prostora
            if previous_pixel_black:
                previous_pixel_black = False
                black_spaces[-1][1] = pixel
    # kao desnu granicu poslednjeg tamnog prostora postavljamo poslednji pixel slike
    black_spaces[-1][1] = width

    for i in black_spaces:
        cv2.circle(img, (i[0], y), 1, (0, 0, 255), thickness=2, lineType=8, shift=0)
        cv2.circle(img, (i[1], y), 1, (0, 255, 0), thickness=2, lineType=8, shift=0)
    cv2.imwrite('Start-end-space-between-2-R-peak.jpg', img)
    #cv2.imshow('Boundaries of space between two P-QRS-T complex', img)


    # u listu boundaries_complex smestamo x koordinate sredina tamnih prostora pronadjenih u prethodnom delu
    # kako bismo oznacili granicu P-QRS-T kompleksa
    # kao levu granicu prvog P-QRS-T kompleksa postavljamo prvi pixel slike
    boundaries_complex.append(0)
    for i in range(1, len(black_spaces)-1):
        boundaries_complex.append(math.ceil((black_spaces[i][1] + black_spaces[i][0])/2))
    # kao desnu granicu poslednjeg P-QRS-T kompleksa postavljamo poslednji pixel slike
    boundaries_complex.append(width)

    img = cv2.imread('K-means-signal.jpg')
    for x in boundaries_complex:
        cv2.circle(img, (x, y), 1, (255, 0, 0), thickness=2, lineType=8, shift=0)
    cv2.imwrite('Boundaries-of-PQRST-complex.jpg', img)
    #cv2.imshow('Boundaries of P-QRS-T complex', img)

    return boundaries_complex



def find_peaks(boundaries, grid_row_height):
    # trazimo tacke na konturi EKG signala koje se nalaze iznad translirane x ose ( x = 1.5 * visina celije mreze)
    height_limit = grid_row_height*1.5

    img = cv2.imread('K-means-signal.jpg')
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 180, 180, 180)
    contours = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)[1]

    areas = []
    for i in range(len(contours)):
        areas.append([cv2.contourArea(contours[i]), i])
    areas = sorted(areas)

    # uzimamo najduzu konturu (najveca povrsina) - konturu signala
    contour_signal = contours[areas[-1][1]]
    cv2.drawContours(img, contours, areas[-1][1], (255, 0, 0), 1)
    #cv2.imshow('Contour of ECG signal', img)
    cv2.imwrite('Contour-ECG.jpg', img)

    # definisemo listu recnika gde je kljuc redni broj P-QRS-T kompleksa, a vrednost je lista uredjenih parova
    # koji predstavljaju koordinate tacaka koje su kandidati za vrh R zupca u posmatranom kompleksu
    coordinates_of_all_peaks = {}
    for i in range(len(boundaries)-1):
        coordinates_of_all_peaks[i+1] = []


    for pair in contour_signal:
        # vrh R zupca trazimo na y koordinati koja je manja od 1.5 * visina celije mreze (kod pravilnog P-QRS-T kompleksa, R zubac ima
        # vrh u intervalu [0, 1/3] mreze, posmatrano sa vrha slike
        if(pair[0][1] <= height_limit):
            # trazimo odgovarajuci P-QRS-T kompleks kom pripada potencijalni vrh R zupca (na osnovu x koordinate tacke)
            for i in range(1, len(boundaries)):
                if(pair[0][0]) in range(boundaries[i-1], boundaries[i]):
                    coordinates_of_all_peaks[i].append([pair[0][1], pair[0][0]])
                    break


    peaks = []
    for complex in coordinates_of_all_peaks:
        # za svaki P-QRS-T kompleks (recnik u listi coordinates) sortiramo listu koordinata tacaka koje su potencijalni vrh R zupca
        peaks_of_one_complex = sorted(coordinates_of_all_peaks[complex])
        # nakon sortiranja biramo uredjeni par na prvom mestu, jer je to tacka cija je y koordinata najmanja - najvisa tacka R zupca
        peaks.append(peaks_of_one_complex[0])


    img = cv2.imread('K-means-signal.jpg')
    for x in peaks:
        cv2.circle(img, (x[1], x[0]), 1, (0, 0, 255), thickness=2, lineType=8, shift=0)
    cv2.imwrite('Peaks-of-PQRST-complex.jpg', img)
    #cv2.imshow('Peaks of all P-QRS-T complex', img)

    return peaks


def calculate_width_R_R_interval(peaks, vertical_lines):
    width_R_R_intervals = []

    for i in range(1, len(peaks)):
        width_R_R_intervals.append((peaks[i][1] - peaks[i-1][1]))

    all_width_equals = True
    low_width = width_R_R_intervals[0]*0.9
    high_width = width_R_R_intervals[0]*1.1
    for width in width_R_R_intervals:
        if width > high_width or width < low_width:
            all_width_equals = False
            break

    if all_width_equals is True:
        # ako su svi R-R intervali priblizno jednake duzine (+- 10%) -> proveravamo da li je u pitanju bradikardija/tahikardija ili ne
        success = check_sin_tachycardia_bradycardia(peaks, vertical_lines)
    else:
        # ako nisu svi R-R intervali priblizno jednake duzine -> proveravamo postojanje aritmije
        success = check_sin_arrhythmia(width_R_R_intervals)

    if success is False:
        print("EKG sa slike se ne može klasifikovati kao EKG na kom je zastupljen neki od obrađivanih poremećaja.")


def check_sin_tachycardia_bradycardia(peaks, vertical_lines_coordinate):
    vertical_lines_coordinate = vertical_lines_coordinate[1:len(vertical_lines_coordinate)-2]

    index_of_start_peak = -1
    index_of_line = -1
    # trazimo vrh nekog od R zubaca koji se poklapa sa nekom od vertikalnih linija (sa greskom od +- 2px)
    # kako bismo pravilno izracunali frekvenciju
    for i in range(0, len(peaks)-1):
        for j in range(0, len(vertical_lines_coordinate)):
            if peaks[i][1] >= (vertical_lines_coordinate[j]-2) and peaks[i][1] <= (vertical_lines_coordinate[j]+2):
                index_of_start_peak = i
                index_of_line = j
                break
        if index_of_start_peak != -1:
            break

    start_peak_x = peaks[index_of_start_peak][1]
    next_peak_x = peaks[index_of_start_peak+1][1]
    distance = next_peak_x - start_peak_x
    column_width = vertical_lines_coordinate[index_of_line+1] - vertical_lines_coordinate[index_of_line]

    number_of_column = 0

    # racunanje frekvencije - prebrojavanje kvadratica na osnovu rastojanja izmedju tekuce pozicije i zavrsnog R zupca
    while(distance >= column_width):
        number_of_column += 1
        distance -= column_width
        index_of_line += 1
        column_width = vertical_lines_coordinate[index_of_line+1] - vertical_lines_coordinate[index_of_line]

    if(number_of_column >= 2 and number_of_column <= 3):
        print("Na testiranom EKG-u je prikazana: SINUSNA TAHIKARDIJA --> frekvencija: [" + str(frequency[number_of_column-1]) + ":" + str(frequency[number_of_column]) + "]")

    elif(number_of_column >= 5):
        print("Na testiranom EKG-u je prikazana: SINUSNA BRADIKARDIJA --> frekvencija: <= " + str(frequency[-1] if number_of_column > len(frequency) else frequency[number_of_column-1]))

    else:
        return False

def check_sin_arrhythmia(width_R_R_intervals):
    width_sorted = sorted(width_R_R_intervals)
    low_limit_bradycardia = 0.9*width_sorted[0]
    high_limit_bradycardia = 1.1*width_sorted[0]
    low_limit_tachycardia = 0.9*width_sorted[-1]
    high_limit_tachycardia = 1.1*width_sorted[-1]

    intervals = []
    # 0 is bradycardia
    # 1 is tachycardia

    success = True
    for width in width_R_R_intervals:
        if(width >= low_limit_bradycardia and width <= high_limit_bradycardia):
            intervals.append(0)
        elif(width >= low_limit_tachycardia and width <= high_limit_tachycardia):
            intervals.append(1)
        else:
            success = False
            break

    if success is False:
        return False
    else:
        error = False
        start_interval = intervals[0]
        already_changed = False
        bradycardia_times = 0
        tachycardia_times = 0
        number_of_bradycardia_intervals = 0
        number_of_tachycardia_intervals = 0
        for i in intervals:
            if i == 0:
                # samo ukoliko je pre pojave bradikardije bio bar jedan P-QRS-T kompleks sa kracom frekvencijom
                # ili EKG zapocinje sa bradikardijom
                # ili je prethodno bila pojava bradikardije
                # moze da se nastavi pracenje toka EKG-a u cilju klasifikovanja poremecaja kao aritmije
                if number_of_bradycardia_intervals > 0 or (start_interval == 0 and already_changed is False) or number_of_tachycardia_intervals >= 1:
                    number_of_tachycardia_intervals = 0
                    already_changed = True
                    if(number_of_bradycardia_intervals == 0):
                        bradycardia_times += 1
                    number_of_bradycardia_intervals += 1
                else:
                    error = True
                    break
            elif i == 1:
                # samo ukoliko je pre pojave tahikardije bilo bar jedan P-QRS-T kompleks sa duzom frekvencijom
                # ili EKG zapocinje sa tahikardijom
                # ili je prethodno bila pojava tahikardije
                # moze da se nastavi pracenje toka EKG-a u cilju klasifikovanja poremecaja kao aritmije
                if number_of_tachycardia_intervals > 0 or (start_interval == 1 and already_changed is False) or number_of_bradycardia_intervals >= 1:
                    number_of_bradycardia_intervals = 0
                    already_changed = True
                    if(number_of_tachycardia_intervals == 0):
                        tachycardia_times += 1
                    number_of_tachycardia_intervals += 1
                else:
                    error = True
                    break

        if error is True:
            return False
        else:
            if (bradycardia_times >= 1 and tachycardia_times > 1) or (bradycardia_times > 1 and tachycardia_times >= 1):
                print("Na testiranom EKG-u je prikazana: SINUSNA ARITMIJA")
            else:
                return False


if __name__ == '__main__':
    file_name = 'sinusna_tahikardija'
    k_means_signal(file_name)
    get_grid(file_name)
    vertical_lines_coordinate = get_vertical_lines()

    grid_cell_width = vertical_lines_coordinate[1] - vertical_lines_coordinate[0]
    boundaries = get_boundaries_of_complex(grid_cell_width)

    peaks = find_peaks(boundaries, grid_cell_width)

    calculate_width_R_R_interval(peaks, vertical_lines_coordinate)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

