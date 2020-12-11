import rticonnextdds_connector as rti



with rti.open_connector("MyParticipantLibrary::MyParticipant","VehicleData.xml") as connector:
    input = connector.get_input("MySubscriber::MyVehicleDataReader")
    input.wait_for_publications()

    while True:
        input.wait()
        input.read()
        for sample in input.samples.valid_data_iter:
            vehicleID = sample.get_number("vehicleID")
            licensePlate = sample.get_string("licensePlate")
            speed = sample.get_number("speed")
            vehicleType = sample.get_string("type")
            print("Recieved Vehicle")
            print("-----------------------------")
            print("VehicleID: " + str(vehicleID))
            print("Licenseplate: " + licensePlate)
            print("Speed: " + str(speed) + " km/h")
            print("Vehicle type: " + vehicleType)
            print("-----------------------------")
            print(" ")
    