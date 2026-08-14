# X2 meshes (fetched from AgiBot — not stored in this repo)

The 45 STL meshes referenced by `x2_ultra.urdf` / `x2_ultra_sphere_feet.urdf`
and the MuJoCo model are AgiBot's official robot-description assets,
byte-identical to their published URDF package:

    https://x2-aimdk.agibot.com/en/latest/_downloads/2ffc9785259556f409e385974a7a0461/X2_URDF-v1.3.0.zip

If that direct link goes stale, the current URDF package is linked from
AgiBot's docs (Get the SDK / Robot Specifications pages):

    https://x2-aimdk.agibot.com/en/latest/get_sdk/index.html
    https://x2-aimdk.agibot.com/en/latest/about_agibot_X2/robot_specifications.html

`install_scripts/setup_x2.sh` downloads and places them here automatically
(`X2_URDF_URL` to override). Manual fallback:

    unzip X2_URDF-v1.3.0.zip && cp X2_URDF-v1.3.0/meshes/*.STL \
        gear_sonic/data/assets/robot_description/urdf/x2_ultra/meshes/

The URDFs in the parent directory are OUR modified variants (sphere-feet
collision, OmniHand integration) and stay in the repo; only the vendor
meshes are fetched.
