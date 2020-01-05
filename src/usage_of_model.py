



# test_images_path = '../test_images/'
# test_image_name = 'Zrzut ekranu z 2019-12-28 14-47-17.png'
# prepared_image = prepare_image(f'{test_images_path}{test_image_name}')
#
# print(prepared_image[0].shape)
#
# prediction = model.predict([prepared_image])
#
# print(prediction)
# np.set_printoptions(suppress=True)  # suppress scientific print
# print(prediction)
#
# index_number = np.argmax(prediction)
# max_confidence = np.max(prediction)
# print(f"Given image is in prediction: {CATEGORIES[index_number]} with {max_confidence * 100}% confidence")