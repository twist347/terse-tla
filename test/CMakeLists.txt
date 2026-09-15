add_library(doctest INTERFACE)
add_library(doctest::doctest ALIAS doctest)
target_include_directories(doctest SYSTEM INTERFACE ${PROJECT_SOURCE_DIR}/thirdparty)

add_executable(tla_test
    main.cpp
    test_vec.cpp
    test_mat.cpp
)
target_link_libraries(tla_test PRIVATE tla::tla doctest::doctest)
target_compile_options(tla_test PRIVATE ${TLA_WARNINGS})

add_test(NAME tla_test COMMAND tla_test)
